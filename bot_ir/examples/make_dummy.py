"""生成 dummy profiling 数据：raw CSV + 入库后的 run 目录，供手工测试。

用法:
    python examples/make_dummy.py [--out run_dummy] [--aliasing] [--hotspot]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from profmw import ProfilingStore
from profmw.dummy import make_dummy_store


def main():
    parser = argparse.ArgumentParser(description="生成 dummy profiling 数据集")
    parser.add_argument("--out", default="run_dummy", help="输出 run 目录")
    parser.add_argument("--aliasing", action="store_true",
                        help="注入 DDR 单通道混叠分布")
    parser.add_argument("--hotspot", action="store_true", help="注入热点核")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    store = make_dummy_store(aliasing=args.aliasing, hotspot=args.hotspot,
                             seed=args.seed)

    # 落一份矩阵形态的 raw CSV（行=模块，列=时间，与真实 raw 数据形态一致），
    # 放在相邻目录避免与 run 目录冲突
    raw_dir = out.with_name(out.name + "_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    for ptype in ("ddr_bw", "core_read_outstanding"):
        store.get_matrix(ptype).to_csv(raw_dir / f"{ptype}.csv")
        print(f"raw CSV   -> {raw_dir / (ptype + '.csv')}（矩阵形态：模块×时间）")

    store.save(out)
    print(f"中间件    -> {out}/（meta.json + types/*.parquet）")
    print("\n验证：重新加载并查询")
    loaded = ProfilingStore.load(out)
    for ptype, meta in loaded.list_types().items():
        df = loaded.get_matrix(ptype)
        print(f"  {ptype}: {df.shape[0]} 模块 × {df.shape[1]} 窗口, unit={meta['unit']}")
    ddr_sum = loaded.aggregate("ddr_bw", t_range=(0.0, 10000.0), agg="sum")
    print(f"  ddr_bw 0~10000 窗口内统计值总和 = {ddr_sum:.1f}")

    # 验证矩阵 CSV 的回读入库路径（真实 raw 数据的接入方式）
    import pandas as pd
    raw_df = pd.read_csv(raw_dir / "ddr_bw.csv", index_col=0)
    re_store = ProfilingStore()
    re_store.add_matrix("ddr_bw", raw_df, meta={"unit": "GB/s"})
    re_sum = re_store.aggregate("ddr_bw", t_range=(0.0, 10000.0), agg="sum")
    assert re_sum == ddr_sum, f"CSV 回读聚合不一致: {re_sum} vs {ddr_sum}"
    print(f"CSV 回读入库后聚合一致: {re_sum:.1f} ✓")


if __name__ == "__main__":
    main()
