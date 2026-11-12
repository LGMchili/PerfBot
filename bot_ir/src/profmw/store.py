"""ProfilingStore：NPU profiling 数据中间件核心。

数据模型：{profiling类型 → DataFrame(index=模块名, columns=升序float时间戳,
值=窗口内基础统计值)} + 两级 meta（run 级 / 类型级）。
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from profmw.validation import validate_store

_META_TYPES_KEY = "types"
_TYPES_DIR = "types"


def _hist_quantile(counts: np.ndarray, edges: np.ndarray, q: float) -> np.ndarray:
    """逐行从直方图计数计算分位数（桶内线性插值），零计数行为 NaN。"""
    total = counts.sum(axis=1)
    out = np.full(counts.shape[0], np.nan)
    m = total > 0
    if not m.any():
        return out
    c, target = counts[m], q * total[m]
    cum = np.cumsum(c, axis=1)
    rows = np.arange(len(c))
    j = (cum >= target[:, None]).argmax(axis=1)  # 第一个累计达到 target 的桶
    prev = np.where(j > 0, cum[rows, j - 1], 0.0)
    inbin = c[rows, j]
    frac = np.clip(
        (target - prev) / np.where(inbin > 0, inbin, 1.0), 0.0, 1.0
    )
    out[m] = edges[j] + frac * (edges[j + 1] - edges[j])
    return out


def _check_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """入库前检查并规整矩阵，返回排序列后的副本。"""
    if not df.index.is_unique:
        dup = df.index[df.index.duplicated()][0]
        raise ValueError(f"行标（模块名）存在重复: {dup!r}")
    try:
        # 先统一列标 dtype（int64、CSV 读入的字符串等归一为 float），
        # 再做重复检查——避免 "100" 与 "100.0" 字符串不同而数值相同的情况漏检
        df = df.set_axis(pd.Index(df.columns).astype("float64"), axis=1)
    except (TypeError, ValueError):
        raise ValueError("列标（时间戳）必须为数值类型") from None
    if not df.columns.is_unique:
        raise ValueError(f"列标（时间戳）存在重复: {df.columns[0]!r}")
    df.index.name = None  # 落盘 CSV 的 "module" 表头读回后归一为无名 index
    return df.sort_index(axis=1)


class ProfilingStore:
    def __init__(self, meta: dict | None = None):
        self.meta: dict = dict(meta or {})
        self._matrices: dict[str, pd.DataFrame] = {}
        self._type_meta: dict[str, dict] = {}

    # ------------------------------------------------------------ 接入

    def add_matrix(self, ptype: str, df: pd.DataFrame, meta: dict | None = None) -> None:
        """接入已为矩阵形态（模块×时间）的窗口统计数据。"""
        if not ptype or "/" in ptype:
            raise ValueError(f"非法类型名: {ptype!r}")
        if ptype in self._matrices:
            raise ValueError(f"类型已存在: {ptype!r}")
        self._matrices[ptype] = _check_matrix(df)
        self._type_meta[ptype] = dict(meta or {})

    def add_type_from_records(
        self,
        ptype: str,
        records: pd.DataFrame,
        window: float | None = None,
        value_col: str = "value",
        time_col: str = "time",
        module_col: str = "module",
        bucket_how: str = "sum",
        meta: dict | None = None,
    ) -> None:
        """接入 raw 事件长表并转为窗口矩阵。

        records 需含时间/模块/数值三列。window=None 时按 records 自带的时间直接
        透视（要求时间已是打点时间）；否则按窗口宽度分桶，列标 = 窗口起点，
        bucket_how ∈ {"sum", "mean"} 决定桶内口径。
        注意：module 或 time 为 NaN 的记录会被静默丢弃（pandas groupby/pivot 行为）。
        """
        if bucket_how not in ("sum", "mean"):
            raise ValueError(f"bucket_how 仅支持 sum/mean，收到: {bucket_how!r}")
        missing = [c for c in (time_col, module_col, value_col) if c not in records.columns]
        if missing:
            raise ValueError(f"records 缺少列: {missing}")
        if window is None:
            df = records.pivot_table(
                index=module_col, columns=time_col, values=value_col, aggfunc="first"
            )
        else:
            if window <= 0:
                raise ValueError(f"window 必须为正数，收到: {window}")
            tmp = pd.DataFrame({
                "module": records[module_col],
                "win": (records[time_col].astype(float) // window) * window,
                "value": records[value_col],
            })
            grouped = tmp.groupby(["module", "win"])["value"].agg(bucket_how)
            df = grouped.unstack("win")
        df = df.astype(float)
        self.add_matrix(ptype, df, meta=meta)

    def add_type_from_histogram(
        self,
        ptype: str,
        records: pd.DataFrame,
        bins,
        quantiles=(0.5, 0.9, 0.99),
        time_col: str = "time",
        module_col: str = "module",
        count_cols: list[str] | None = None,
        meta: dict | None = None,
    ) -> list[str]:
        """硬件直方图计数 -> 每个分位数一张标准矩阵（分位数展开存储）。

        records 每行一个 (模块, 窗口) 快照：时间/模块列 + 每桶一列计数，
        count 列顺序须与 bins 一一对应（默认取其余列的文件顺序）。
        生成类型名 f"{ptype}.p{分位数*100}"（如 dma_lat.p50 / dma_lat.p99.9），
        作为普通矩阵入库，后续 save/load/聚合/报告零改动复用；
        meta 追加 quantile 与 base_type 字段。零计数的 (模块, 窗口) 为 NaN。
        """
        edges = np.asarray(bins, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError(f"bins 必须为严格递增的桶边界数组，收到: {bins!r}")
        quantiles = tuple(quantiles)
        for q in quantiles:
            if not 0.0 < q <= 1.0:
                raise ValueError(f"quantile 必须在 (0, 1] 内，收到: {q}")
        if count_cols is None:
            count_cols = [c for c in records.columns if c not in (time_col, module_col)]
        if len(count_cols) != len(edges) - 1:
            raise ValueError(
                f"count 列数 {len(count_cols)} 应等于桶数 {len(edges) - 1}"
                "（count 列顺序须与 bins 对应）"
            )
        missing = [c for c in (time_col, module_col, *count_cols) if c not in records.columns]
        if missing:
            raise ValueError(f"records 缺少列: {missing}")
        counts = records[count_cols].to_numpy(dtype=float)
        created = []
        for q in quantiles:
            vals = _hist_quantile(counts, edges, q)
            long = pd.DataFrame({
                module_col: records[module_col].to_numpy(),
                time_col: records[time_col].astype(float).to_numpy(),
                "value": vals,
            })
            wide = long.pivot_table(index=module_col, columns=time_col,
                                    values="value", aggfunc="first", dropna=False)
            qname = f"{ptype}.p{q * 100:g}"
            self.add_matrix(qname, wide.astype(float),
                            meta={**(meta or {}), "quantile": q, "base_type": ptype})
            created.append(qname)
        return created

    # ------------------------------------------------------------ 查询

    def list_types(self) -> dict[str, dict]:
        return {name: dict(m) for name, m in self._type_meta.items()}

    def get_matrix(self, ptype: str) -> pd.DataFrame:
        if ptype not in self._matrices:
            raise KeyError(f"未知 profiling 类型: {ptype!r}，现有: {list(self._matrices)}")
        return self._matrices[ptype].copy()

    def window_series(
        self,
        ptype: str,
        modules: list[str] | None = None,
        module_prefix: str | None = None,
        t_range: tuple[float | None, float | None] | None = None,
        skipna: bool = True,
        interp: str | None = None,
    ) -> pd.DataFrame:
        """取满足条件的逐窗口切片（模块×时间，聚合前）。"""
        return self._select(ptype, modules, module_prefix, t_range, interp)

    def aggregate(
        self,
        ptype: str,
        modules: list[str] | None = None,
        module_prefix: str | None = None,
        t_range: tuple[float | None, float | None] | None = None,
        agg: str = "sum",
        skipna: bool = True,
        interp: str | None = None,
    ) -> float:
        """对选中的 (模块×窗口) 单元格沿时间轴聚合，返回单个观测值。

        agg ∈ sum/mean/var/max/min/count。var 为样本方差(ddof=1)。
        skipna=True 跳过 NaN；False 时 NaN 参与运算并传播
        （count 例外：skipna=False 时计入 NaN，即返回单元格总数）。
        """
        if agg not in ("sum", "mean", "var", "max", "min", "count"):
            raise ValueError(f"不支持的 agg: {agg!r}")
        df = self._select(ptype, modules, module_prefix, t_range, interp)
        arr = df.to_numpy(dtype=float).ravel()
        if skipna:
            valid = arr[~np.isnan(arr)]
        else:
            valid = arr
        if agg == "sum":
            return float(np.nansum(valid)) if skipna else float(valid.sum())
        if agg == "mean":
            return float(np.nanmean(valid)) if skipna else float(valid.mean())
        if agg == "var":
            return float(np.nanvar(valid, ddof=1)) if skipna else float(valid.var(ddof=1))
        if agg == "max":
            return float(np.nanmax(valid)) if skipna else float(valid.max())
        if agg == "min":
            return float(np.nanmin(valid)) if skipna else float(valid.min())
        return int((~np.isnan(arr)).sum()) if skipna else int(arr.size)  # count

    def _select(self, ptype, modules, module_prefix, t_range, interp) -> pd.DataFrame:
        """模块/前缀/时间范围过滤 + 可选边界插值，返回切片副本。"""
        if interp is not None and interp != "linear":
            raise ValueError(f"不支持的 interp: {interp!r}（仅支持 'linear'）")
        df = self._matrices[ptype] if ptype in self._matrices else None
        if df is None:
            raise KeyError(f"未知 profiling 类型: {ptype!r}，现有: {list(self._matrices)}")
        if modules is None:
            rows = list(df.index)
        else:
            missing = [m for m in modules if m not in df.index]
            if missing:
                raise KeyError(f"未知模块: {missing}，现有: {list(df.index)}")
            rows = modules
        if module_prefix is not None:
            rows = [m for m in rows if m.startswith(module_prefix)]
        if not rows:
            raise ValueError("过滤后无任何模块")
        out = df.loc[rows]
        if t_range is not None:
            t0, t1 = t_range
            if t0 is not None and t1 is not None and t0 > t1:
                raise ValueError(f"t_range 起点大于终点: {t_range}")
            cols = df.columns.astype(float)
            mask = np.ones(len(cols), dtype=bool)
            if t0 is not None:
                mask &= cols >= t0
            if t1 is not None:
                mask &= cols <= t1
            inside = out.loc[:, mask]
            if interp is None:
                selected = inside
            else:
                left = self._boundary(df, cols, rows, t0) if t0 is not None and (t0 not in cols) else None
                right = self._boundary(df, cols, rows, t1) if t1 is not None and (t1 not in cols) else None
                if left is not None and right is not None and t0 == t1:
                    right = None  # 退化区间：只保留一个虚拟边界点
                parts = [p for p in (left, inside, right) if p is not None and len(p.columns)]
                selected = pd.concat(parts, axis=1) if parts else inside.iloc[:, :0]
            if selected.shape[1] == 0:
                lo, hi = (cols[0], cols[-1]) if len(cols) else ("-", "-")
                raise ValueError(
                    f"时间范围 {t_range} 未覆盖任何打点时间（数据范围 {lo}~{hi}）"
                )
            return selected.copy()
        return out.copy()

    @staticmethod
    def _boundary(df: pd.DataFrame, cols: np.ndarray, rows, t: float) -> pd.DataFrame | None:
        """t 落在两列之间时返回插值虚拟列；超出数据范围或无左邻时返回 None。"""
        pos = np.searchsorted(cols, t)
        if pos == 0 or pos >= len(cols):  # 超出首末列范围，不外推
            return None
        t_lo, t_hi = float(cols[pos - 1]), float(cols[pos])
        vals = df.loc[rows]
        v_lo = vals.iloc[:, pos - 1].to_numpy(dtype=float)
        v_hi = vals.iloc[:, pos].to_numpy(dtype=float)
        frac = (t - t_lo) / (t_hi - t_lo)
        with np.errstate(invalid="ignore"):
            interped = v_lo + (v_hi - v_lo) * frac
        interped[np.isnan(v_lo) | np.isnan(v_hi)] = np.nan  # 邻居含 NaN 不插值
        return pd.DataFrame({t: interped}, index=rows)

    # ------------------------------------------------------------ 校验

    def validate(self, nan_ratio_warn: float = 0.5):
        """数据健康检查，返回 ValidationReport。"""
        return validate_store(self, nan_ratio_warn=nan_ratio_warn)

    # ------------------------------------------------------------ IO

    def save(self, path, zip_pack: bool = False) -> None:
        """保存为目录（path/ 下 meta.json + types/*.csv）；zip_pack=True 时打包为 path.zip。"""
        path = Path(path)
        target = path.parent / (path.name + ".zip") if zip_pack else path
        if target.exists():
            raise FileExistsError(f"目标已存在: {target}")
        if zip_pack:
            with tempfile.TemporaryDirectory() as tmp:
                self._save_dir(Path(tmp))
                shutil.make_archive(str(path), "zip", root_dir=tmp)
        else:
            self._save_dir(path)

    def _save_dir(self, path: Path) -> None:
        (path / _TYPES_DIR).mkdir(parents=True, exist_ok=True)
        for ptype, df in self._matrices.items():
            out = df.copy()
            out.columns = [repr(float(c)) for c in df.columns]  # repr(float) 往返精确
            out.index.name = "module"
            out.to_csv(path / _TYPES_DIR / f"{ptype}.csv")
        meta = {"run": self.meta, _META_TYPES_KEY: self._type_meta}
        (path / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path) -> "ProfilingStore":
        path = Path(path)
        if path.suffix == ".zip":
            if not path.exists():
                raise FileNotFoundError(f"文件不存在: {path}")
            with tempfile.TemporaryDirectory() as tmp:
                shutil.unpack_archive(str(path), extract_dir=tmp)
                return cls._load_dir(Path(tmp))
        if not path.is_dir():
            raise FileNotFoundError(f"目录不存在: {path}")
        return cls._load_dir(path)

    @classmethod
    def _load_dir(cls, path: Path) -> "ProfilingStore":
        meta_file = path / "meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"缺少 meta.json: {meta_file}")
        raw = json.loads(meta_file.read_text(encoding="utf-8"))
        store = cls(meta=raw.get("run", {}))
        types_dir = path / _TYPES_DIR
        for ptype, tmeta in raw.get(_META_TYPES_KEY, {}).items():
            f = types_dir / f"{ptype}.csv"
            if not f.exists():
                raise FileNotFoundError(f"类型 {ptype!r} 的数据文件缺失: {f}")
            df = pd.read_csv(f, index_col=0)
            store._matrices[ptype] = _check_matrix(df)
            store._type_meta[ptype] = dict(tmeta)
        return store
