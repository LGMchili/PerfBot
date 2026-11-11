"""profmw 演示：构建矩阵数据集 -> 按项目格式保存 -> 从保存目录读取 -> 求窗口统计值 -> HTML 报告。

用法:
    python examples/demo.py          # 需要生成报告时先 pip install plotly
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from profmw import ProfilingStore
from profmw.report import export_report

OUT = Path("run_demo")
REPORT = Path("run_demo_report.html")

TIMES = np.arange(0.0, 5000.0, 100.0)  # 50 个打点时间，列标


def build_matrix(rows: list[str], values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(values, index=rows, columns=TIMES)


def build_types() -> list[tuple[str, pd.DataFrame, dict]]:
    """三组矩阵数据：DDR 带宽 / 96 通道 rx_req 平均深度 / 120 核 outstanding。"""
    rng = np.random.default_rng(42)
    out = []

    # 1. ddr_bw: 8 通道带宽（GB/s），CH2 注入高负载
    bw = rng.uniform(20.0, 30.0, size=(8, len(TIMES)))
    bw[2] = rng.uniform(90.0, 100.0, size=len(TIMES))
    out.append(("ddr_bw", build_matrix([f"CH{i}" for i in range(8)], bw),
                {"unit": "GB/s", "stat": "窗口内均值", "description": "DDR 通道带宽"}))

    # 2. rx_req_avg_depth: 96 通道 RX 请求平均队列深度，多数通道 2~8，
    #    CH00/CH47/CH95 注入积压（深度 40~60）——热力图上呈亮条
    depth = rng.uniform(2.0, 8.0, size=(96, len(TIMES)))
    depth[[0, 47, 95]] = rng.uniform(40.0, 60.0, size=(3, len(TIMES)))
    out.append(("rx_req_avg_depth",
                build_matrix([f"CH{i:02d}" for i in range(96)], depth),
                {"unit": "count", "stat": "窗口内平均深度",
                 "description": "各通道 RX 请求平均队列深度"}))

    # 3. core_read_outstanding: 120 核读 outstanding，多数核 4~16，
    #    热点核 CORE007/CORE119 注入 60~80
    osts = rng.uniform(4.0, 16.0, size=(120, len(TIMES)))
    osts[[7, 119]] = rng.uniform(60.0, 80.0, size=(2, len(TIMES)))
    out.append(("core_read_outstanding",
                build_matrix([f"CORE{i:03d}" for i in range(120)], osts),
                {"unit": "count", "stat": "窗口内均值",
                 "description": "各核窗口内平均读 outstanding 数"}))
    return out


def main():
    print("== 1. 构建矩阵数据集（行=模块，列=float 打点时间）==")
    store = ProfilingStore(meta={"benchmark": "demo", "chip": "X9-v0.9", "date": "2026-11-11"})
    for ptype, df, meta in build_types():
        store.add_matrix(ptype, df, meta=meta)
        print(f"  {ptype}: {df.shape[0]} 模块 × {df.shape[1]} 时间点, unit={meta['unit']}")

    print("\n== 2. 按项目格式保存（meta.json + types/*.csv）==")
    if OUT.exists():
        shutil.rmtree(OUT)
    store.save(OUT)
    for f in sorted(OUT.rglob("*")):
        print(f"  {f}")

    print("\n== 3. 从保存的目录读取 ==")
    loaded = ProfilingStore.load(OUT)
    print("  run meta :", loaded.meta)
    print("  类型     :", {t: m["unit"] for t, m in loaded.list_types().items()})

    print("\n== 4. 求窗口内统计值 ==")
    print("  ddr_bw  [0, 5000) 全通道总和 =",
          round(loaded.aggregate("ddr_bw", agg="sum"), 1))
    print("  rx_req  全时段最大深度 =",
          round(loaded.aggregate("rx_req_avg_depth", agg="max"), 1))
    print("  rx_req  CH00 均值（积压通道） =",
          round(loaded.aggregate("rx_req_avg_depth", modules=["CH00"], agg="mean"), 1))
    print("  rx_req  其余通道均值 =",
          round(loaded.aggregate("rx_req_avg_depth", module_prefix="CH",
                                 modules=[f"CH{i:02d}" for i in range(1, 95)],
                                 agg="mean"), 1))
    print("  core    热点核均值 =",
          round(loaded.aggregate("core_read_outstanding",
                                 modules=["CORE007", "CORE119"], agg="mean"), 1))
    print("  core    [1550, 2450) 边界插值 min =",
          round(loaded.aggregate("core_read_outstanding", t_range=(1550.0, 2450.0),
                                 agg="min", interp="linear"), 1))

    print("\n== 5. 校验 ==")
    print(loaded.validate())

    print("\n== 6. 生成 HTML 观测报告（热力图/波形/排名/统计表）==")
    out = export_report(loaded, REPORT)
    print(f"  {out}（双击打开，离线可用）")


if __name__ == "__main__":
    main()
