"""profmw 演示：构建矩阵数据集 + NoC 接入 -> 保存 -> 读取 -> 窗口统计 -> HTML 报告。

所有产物统一落进 run_demo/：中间件本体（meta.json + types/*.csv）与
NoC raw 契约示例（raw/ 子目录，manifest + 指标矩阵 CSV）；波形报告与
ring矩阵报告分开生成（ring 报告为纯 DOM，无需 plotly）。

用法:
    python examples/demo.py          # 未装 plotly 时跳过波形报告，ring 报告照常生成
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from profmw import ProfilingStore
from profmw.dummy import make_noc_raw_dir
from profmw.noc import add_noc_raw
from profmw.report import export_report, export_ring_report

OUT = Path("run_demo")
REPORT = Path("run_demo_report.html")
RING_REPORT = Path("run_demo_ring_report.html")

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
                {"module": "dummy", "unit": "GB/s", "stat": "窗口内均值",
                 "description": "DDR 通道带宽"}))

    # 2. rx_req_avg_depth: 96 通道 RX 请求平均队列深度，CH00/CH47/CH95 注入积压
    depth = rng.uniform(2.0, 8.0, size=(96, len(TIMES)))
    depth[[0, 47, 95]] = rng.uniform(40.0, 60.0, size=(3, len(TIMES)))
    out.append(("rx_req_avg_depth",
                build_matrix([f"CH{i:02d}" for i in range(96)], depth),
                {"module": "dummy", "unit": "count", "stat": "窗口内平均深度",
                 "description": "各通道 RX 请求平均队列深度"}))

    # 3. core_read_outstanding: 120 核读 outstanding，热点核注入 60~80
    osts = rng.uniform(4.0, 16.0, size=(120, len(TIMES)))
    osts[[7, 119]] = rng.uniform(60.0, 80.0, size=(2, len(TIMES)))
    out.append(("core_read_outstanding",
                build_matrix([f"CORE{i:03d}" for i in range(120)], osts),
                {"module": "dummy", "unit": "count", "stat": "窗口内均值",
                 "description": "各核窗口内平均读 outstanding 数"}))
    return out


def main():
    print("== 1. 构建矩阵数据集（行=模块，列=float 打点时间）==")
    store = ProfilingStore(meta={"benchmark": "demo", "chip": "X9-v0.9", "date": "2026-11-16"})
    for ptype, df, meta in build_types():
        store.add_matrix(ptype, df, meta=meta)
        print(f"  {ptype}: {df.shape[0]} 模块 × {df.shape[1]} 时间点, unit={meta['unit']}")

    print("\n== 2. NoC ring raw 接入（manifest + 每指标一张矩阵 CSV）==")
    with tempfile.TemporaryDirectory() as tmp:   # scratch：run 结束后不留额外目录
        raw = make_noc_raw_dir(Path(tmp) / "noc_raw")   # 8 ring × 4 通道 × 25 CS，R03_C02 CS12~14 注入环段拥塞
        add_noc_raw(store, raw)                  # 校验行标约定，levels 写入 run meta
        for ptype in ("noc_up_fail_count", "noc_left_util"):
            print(f"  {ptype}: 800 模块 × 200 时间点（已入库）")

        print("\n== 3. 按项目格式保存（meta.json + types/*.csv + raw/）==")
        if OUT.exists():
            shutil.rmtree(OUT)
        store.save(OUT)
        shutil.move(str(raw), str(OUT / "raw"))  # raw 契约示例归档进 run 目录
    files = sorted(f for f in OUT.rglob("*") if f.is_file())
    print(f"  {OUT}/（{len(files)} 个文件，含 raw/ 子目录的契约示例）")

    print("\n== 4. 从保存的目录读取 ==")
    loaded = ProfilingStore.load(OUT)
    print("  run meta :", loaded.meta)
    print("  类型     :", {t: m["unit"] for t, m in loaded.list_types().items()})

    print("\n== 5. 求窗口内统计值 ==")
    print("  ddr_bw  [0, 5000) 全通道总和 =",
          round(loaded.aggregate("ddr_bw", agg="sum"), 1))
    print("  rx_req  全时段最大深度 =",
          round(loaded.aggregate("rx_req_avg_depth", agg="max"), 1))
    print("  core    热点核均值 =",
          round(loaded.aggregate("core_read_outstanding",
                                 modules=["CORE007", "CORE119"], agg="mean"), 1))
    print("  core    [1550, 2450) 边界插值 min =",
          round(loaded.aggregate("core_read_outstanding", t_range=(1550.0, 2450.0),
                                 agg="min", interp="linear"), 1))
    print("  --- NoC 层级查询（前缀 = 层级）---")
    print("  拥塞注入点 R03_C02_CS13 失败计数均值 =",
          round(loaded.aggregate("noc_up_fail_count",
                                 modules=["R03_C02_CS13"], agg="mean"), 1))
    print("  R01_ 整条 ring 失败计数总和 =",
          round(loaded.aggregate("noc_up_fail_count", module_prefix="R01_", agg="sum")))
    print("  --- 分维聚合 magg/tagg ---")
    print("  R01 ring 平均每窗口总失败数 (magg=sum, tagg=mean) =",
          round(loaded.aggregate("noc_up_fail_count", module_prefix="R01_",
                                 magg="sum", tagg="mean"), 1))
    print("  R03_C02 通道逐窗口失败总数曲线 (magg=sum) 前5点 =",
          list(loaded.aggregate("noc_up_fail_count", module_prefix="R03_C02_",
                                magg="sum").iloc[:5].round(0)))

    print("\n== 6. 校验 ==")
    print(loaded.validate())

    print("\n== 7. 生成 HTML 观测报告（波形报告 + ring矩阵报告，分开生成）==")
    try:
        out = export_report(loaded, REPORT)
        print(f"  波形报告: {out}（波形/热力图/统计表）")
    except ImportError:
        print("  跳过波形报告：未安装 plotly。pip install plotly 后重跑即可生成")
    ring = export_ring_report(loaded, RING_REPORT)
    print(f"  ring矩阵报告: {ring}（纯 DOM，无需 plotly）")


if __name__ == "__main__":
    main()
