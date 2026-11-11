"""profmw 演示：构建矩阵数据 -> 按项目格式保存 -> 从保存目录读取 -> 求窗口统计值。

用法:
    python examples/demo.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from profmw import ProfilingStore

OUT = Path("run_demo")


def main():
    print("== 1. 构建矩阵数据（8 个 DDR 通道 × 50 个打点时间）==")
    times = np.arange(0.0, 5000.0, 100.0)  # 列标：float 打点时间
    rng = np.random.default_rng(42)
    data = rng.uniform(20.0, 30.0, size=(8, len(times)))  # 正常负载 20~30 GB/s
    data[2] = rng.uniform(90.0, 100.0, size=len(times))   # CH2 注入高负载
    df = pd.DataFrame(data, index=[f"CH{i}" for i in range(8)], columns=times)
    print(df.iloc[:3, :5].round(1).to_string(), "\n...")

    store = ProfilingStore(meta={"benchmark": "demo", "chip": "X9-v0.9", "date": "2026-11-11"})
    store.add_matrix("dram_bw", df,
                     meta={"unit": "GB/s", "stat": "窗口内均值", "description": "DDR 通道带宽"})

    print("\n== 2. 按项目格式保存（meta.json + types/*.csv）==")
    if OUT.exists():
        shutil.rmtree(OUT)
    store.save(OUT)
    for f in sorted(OUT.rglob("*")):
        print(f"  {f}")

    print("\n== 3. 从保存的目录读取 ==")
    loaded = ProfilingStore.load(OUT)
    print("  run meta :", loaded.meta)
    print("  类型 meta:", loaded.list_types())

    print("\n== 4. 求窗口内统计值 ==")
    print("  [0, 5000) 全通道统计值总和 =",
          loaded.aggregate("dram_bw", agg="sum"))
    print("  [0, 1000) CH2 峰值 =",
          loaded.aggregate("dram_bw", modules=["CH2"], t_range=(0.0, 1000.0), agg="max"))
    print("  [0, 1000) 其余通道峰值 =",
          loaded.aggregate("dram_bw", modules=[f"CH{i}" for i in (0, 1, 3, 4, 5, 6, 7)],
                           t_range=(0.0, 1000.0), agg="max"))
    print("  CH2 全时段方差 =",
          round(loaded.aggregate("dram_bw", modules=["CH2"], agg="var"), 2))
    print("  边界插值：[1550, 2450) 内 CH2 的 min =",
          loaded.aggregate("dram_bw", modules=["CH2"], t_range=(1550.0, 2450.0),
                           agg="min", interp="linear"))

    print("\n  CH0~CH2 在 [1550, 2450) 的逐窗口序列（首末列为插值边界点）：")
    print(loaded.window_series("dram_bw", modules=["CH0", "CH1", "CH2"],
                               t_range=(1550.0, 2450.0), interp="linear").round(2).to_string())

    print("\n== 5. 校验 ==")
    print(loaded.validate())


if __name__ == "__main__":
    main()
