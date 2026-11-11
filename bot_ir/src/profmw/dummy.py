"""Dummy 数据生成：模拟 NPU profiling 形态的测试数据。

- ddr_bw：96 个 DDR 通道带宽（CH00~CH95），可选"单通道混叠"病态分布
  （少数通道负载极高、其余接近零，复刻 case-003 的 [99, 1.5, 1, 0.5] 形态）；
- core_read_outstanding：120 个核的读 outstanding 统计（CORE000~CORE119），
  可选热点核模式。
"""

from __future__ import annotations

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
