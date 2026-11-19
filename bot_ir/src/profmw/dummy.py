"""Dummy 数据生成：模拟 NPU profiling 形态的测试数据。

- ddr_bw：96 个 DDR 通道带宽（CH00~CH95），可选"单通道混叠"病态分布
  （少数通道负载极高、其余接近零，复刻 case-003 的 [99, 1.5, 1, 0.5] 形态）；
- core_read_outstanding：120 个核的读 outstanding 统计（CORE000~CORE119），
  可选热点核模式；
- NoC ring：8 ring × 4 通道（Req/Snp/Rsp/Dat）× 25 CS
  （R00_Req_CS00 ~ R07_Dat_CS24），可选注入 R03_Rsp 通道 CS12~CS14 环段拥塞。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from profmw.store import ProfilingStore

WINDOW = 100.0        # 窗口宽度（时间单位自定，如 µs）
N_WINDOWS = 200       # 默认窗口数
DDR_CHANNELS = 96
CORES = 120

_SPECS = {
    "ddr_bw": dict(
        names=[f"CH{i:02d}" for i in range(DDR_CHANNELS)],
        lo=20.0, hi=40.0,            # 正常：每通道 20~40 GB/s
        unit="GB/s",
        description="各 DDR 通道窗口内平均带宽",
    ),
    "core_read_outstanding": dict(
        names=[f"CORE{i:03d}" for i in range(CORES)],
        lo=4.0, hi=16.0,             # 正常：每核 4~16 个 outstanding 读
        unit="count",
        description="各核窗口内平均读 outstanding 数",
    ),
}


def _matrix_values(ptype: str, rng: np.random.Generator, anomaly: bool,
                   n_windows: int) -> np.ndarray:
    spec = _SPECS[ptype]
    n = len(spec["names"])
    if not anomaly:
        return rng.uniform(spec["lo"], spec["hi"], size=(n, n_windows))
    vals = rng.uniform(0.2, 2.0, size=(n, n_windows))  # 冷模块接近零
    hot = rng.choice(n, size=max(4, n // 24), replace=False)
    vals[hot] = rng.uniform(80.0, 99.0, size=(len(hot), n_windows))
    return vals


def make_dummy_records(
    ptype: str,
    n_windows: int = N_WINDOWS,
    window: float = WINDOW,
    anomaly: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """生成某类型的 raw 事件长表（time/module/value），供 add_type_from_records 使用。"""
    if ptype not in _SPECS:
        raise ValueError(f"未知 dummy 类型: {ptype!r}，现有: {list(_SPECS)}")
    rng = np.random.default_rng(seed)
    spec = _SPECS[ptype]
    n = len(spec["names"])
    values = _matrix_values(ptype, rng, anomaly, n_windows)
    # 每个窗口内取一个随机偏移时刻作为打点时间
    offsets = rng.uniform(0.0, window * 0.9, size=(n, n_windows))
    times = np.broadcast_to(np.arange(n_windows) * window, (n, n_windows)) + offsets
    return pd.DataFrame(
        {
            "time": times.ravel(),
            "module": np.repeat(spec["names"], n_windows),
            "value": values.ravel(),
        }
    ).sort_values("time").reset_index(drop=True)


def make_dummy_store(
    aliasing: bool = False,
    hotspot: bool = False,
    n_windows: int = N_WINDOWS,
    seed: int = 0,
) -> ProfilingStore:
    """生成填好 dummy 数据的 ProfilingStore（经 raw 接入全流程）。

    aliasing=True 时 ddr_bw 注入单通道混叠分布；hotspot=True 时
    core_read_outstanding 注入热点核。
    """
    store = ProfilingStore(
        meta={"benchmark": "dummy", "chip": "X9-dummy", "date": "2026-11-11"}
    )
    for ptype, anomaly, seed_off in (
        ("ddr_bw", aliasing, 0),
        ("core_read_outstanding", hotspot, 1),
    ):
        spec = _SPECS[ptype]
        records = make_dummy_records(
            ptype, n_windows=n_windows, anomaly=anomaly, seed=seed + seed_off
        )
        store.add_type_from_records(
            ptype, records, window=WINDOW, bucket_how="mean",
            meta={
                "unit": spec["unit"],
                "stat": "窗口内均值",
                "description": spec["description"],
                "source": "profmw.dummy",
            },
        )
    return store


# ---------------------------------------------------------------- NoC ring 生成

NOC_RINGS = 8
NOC_CHANNELS = ("Req", "Snp", "Rsp", "Dat")
NOC_CS = 25

_NOC_META = {
    "noc_up_fail_count": {"module": "dummy", "unit": "count",
                          "stat": "窗口内失败次数",
                          "description": "各节点上 ring 失败计数"},
    "noc_down_fail_count": {"module": "dummy", "unit": "count",
                            "stat": "窗口内失败次数",
                            "description": "各节点下 ring 失败计数"},
    "noc_left_util": {"module": "dummy", "unit": "%",
                      "stat": "窗口内平均占用率",
                      "description": "环左方向链路占用率"},
    "noc_right_util": {"module": "dummy", "unit": "%",
                       "stat": "窗口内平均占用率",
                       "description": "环右方向链路占用率"},
}


def make_noc_raw_dir(out_dir, anomaly: bool = True, seed: int = 0,
                     n_windows: int = N_WINDOWS) -> Path:
    """生成 noc_raw 目录（manifest.json + 每指标一张矩阵 CSV）。

    布局：8 ring × 4 通道（Req/Snp/Rsp/Dat）× 25 CS = 800 行/指标。anomaly=True
    时在 R03_Rsp 通道的 CS12~CS14 注入环段拥塞（失败计数高、左右占用率近饱和）。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    times = np.arange(n_windows) * WINDOW
    modules = [f"R{r:02d}_{ch}_CS{s:02d}"
               for r in range(NOC_RINGS)
               for ch in NOC_CHANNELS
               for s in range(NOC_CS)]
    n = len(modules)
    mats = {
        "noc_up_fail_count": rng.integers(0, 4, (n, n_windows)).astype(float),
        "noc_down_fail_count": rng.integers(0, 4, (n, n_windows)).astype(float),
        "noc_left_util": rng.uniform(20.0, 60.0, (n, n_windows)),
        "noc_right_util": rng.uniform(20.0, 60.0, (n, n_windows)),
    }
    if anomaly:
        for i, m in enumerate(modules):
            if m.startswith("R03_Rsp_CS") and int(m[-2:]) in (12, 13, 14):
                mats["noc_up_fail_count"][i] = rng.integers(80, 121, n_windows)
                mats["noc_down_fail_count"][i] = rng.integers(60, 101, n_windows)
                mats["noc_left_util"][i] = rng.uniform(88.0, 99.0, n_windows)
                mats["noc_right_util"][i] = rng.uniform(88.0, 99.0, n_windows)
    manifest = {}
    for ptype, arr in mats.items():
        pd.DataFrame(arr, index=modules, columns=times).to_csv(
            out / f"{ptype}.csv")
        manifest[ptype] = dict(_NOC_META[ptype])
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
