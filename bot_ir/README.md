# bot_ir — NPU Profiling 数据中间件

NPU 性能分析框架的 **L0 IR 层数据中间件**（对应《NPU-Agent-Profiling 第一性原理分析》3.2 节）：
把 raw profiling 数据封装成带标签的二维矩阵，向上层（L2 诊断 Agent 等）提供廉价、确定性的
统计查询服务。所有数据加工、算术、验证在本层完成，LLM 只负责提议查询、不直接算数。

## 数据模型

```
ProfilingStore = { profiling类型 → 二维矩阵 } + 两级 meta
```

- **行标**：模块名，平铺字符串（如 `CH00`、`CORE017`）
- **列标**：打点时间，float，升序；各类型时间轴彼此独立（采样率不同不要求对齐）
- **矩阵值**：窗口内基础统计值（口径由写入方在 meta 中声明）
- **run meta**：benchmark、芯片版本、采集日期等
- **类型 meta**：单位（unit）、统计口径（stat）、描述、来源

数据规模定位：每类型 <1000 模块 × <100 万打点均可，典型场景（<100 × 1 万）下聚合查询毫秒级。

## 落盘格式（单 run 单目录）

```
run_case003/
├── meta.json              # run meta + 每类型 meta，人类可读
└── types/
    └── dram_bw.csv        # module,0.0,100.0,200.0,...  行=模块，列=时间戳
```

矩阵即普通 CSV（首列 `module`），与 raw 数据同构，可肉眼检查、可被任何工具读取；
列标用 `repr(float)` 写出，读回精确无损。手工摆一个这种目录（不经 save）也能直接 load。
`save(path, zip_pack=True)` 可打包为 `path.zip` 单文件分发。

## 安装

唯一重依赖为 pandas：

```bash
pip install -e bot_ir          # 或在 bot_ir/ 内: pip install -e .
```

## 快速上手

### 1. 接入数据（矩阵形态）

```python
import pandas as pd
from profmw import ProfilingStore

df = pd.read_csv("real_raw.csv", index_col=0)      # 首列模块名作为行标
store = ProfilingStore(meta={"benchmark": "bw_gather", "chip": "X9-v0.9"})
store.add_matrix("dram_bw", df, meta={"unit": "GB/s", "stat": "窗口内均值"})
# 列标字符串/int64 会被自动归一为 float64；重复行标/列标会报错
```

> 备用接口：`add_type_from_records` 支持逐事件长表（time/module/value 三列，
> `window=` 分桶、`bucket_how=` 定口径后转矩阵），供将来接入厂商逐事件导出使用，
> 日常矩阵数据不经过它。详见 docstring。

**延时分布（硬件直方图计数）**：raw 为每 (模块, 窗口) 一行的直方图计数快照时，
按分位数展开为若干张标准矩阵入库：

```python
hist = pd.read_csv("dma_lat_hist.csv")   # time/module 列 + 每桶一列计数（顺序对应 bins）
store.add_type_from_histogram("dma_lat", hist, bins=[0, 32, 64, 128, 256, 512],
                              quantiles=(0.5, 0.9, 0.99), meta={"unit": "ns"})
# 生成 dma_lat.p50 / dma_lat.p90 / dma_lat.p99 三张普通矩阵：
# 零计数 (模块, 窗口) 为 NaN；此后 save/load/聚合/热力图/HTML 报告零改动复用。
# 分位数在桶内线性插值；raw 直方图留在 raw 层归档，需要新分位数时重新 ingest 即可。
```

### 2. 保存 / 加载

```python
store.save("run_case003/")                 # 目录；已存在会拒绝，防覆盖
store.save("run_case003", zip_pack=True)   # 生成 run_case003.zip
loaded = ProfilingStore.load("run_case003/")
loaded.list_types()                        # {"dram_bw": {"unit": "GB/s", ...}}
loaded.get_matrix("dram_bw")               # 返回矩阵副本
```

### 3. 聚合查询（性能观测值现算，不落盘）

```python
store.aggregate("dram_bw", agg="sum")                                # 全模块全时段总和
store.aggregate("dram_bw", modules=["CH0", "CH1"],
                t_range=(1000.0, 5000.0), agg="max")                 # 指定模块+时间范围
store.aggregate("dram_bw", module_prefix="CORE",
                t_range=(None, 5000.0), agg="var")                   # 前缀过滤+开端时间
```

- `agg` ∈ `sum / mean / var / max / min / count`；`var` 为样本方差（ddof=1）
- `t_range=(t0, t1)`，任一端可为 `None`（开放区间）；只支持范围查询，不支持单点等值取值
- 缺失值为 NaN，默认 `skipna=True`（`False` 时 NaN 参与运算并传播；count 例外，见 docstring）
- 时间范围超出数据范围、t0>t1 均报带上下文的 ValueError

**逐窗口序列**（聚合前的切片，供画图/进一步分析）：

```python
store.window_series("dram_bw", modules=["CH2"], t_range=(1550.0, 2450.0))
```

### 4. 边界插值

查询边界不在打点时间上时，`interp="linear"` 由相邻两列插出边界虚拟值：

```python
store.aggregate("dram_bw", t_range=(1550.0, 2450.0), agg="min", interp="linear")
store.window_series("dram_bw", t_range=(1550.0, 2450.0), interp="linear")
# 首末列为虚拟边界点；不外推（超出数据范围不产生数据）；
# 某模块相邻列含 NaN 时该模块边界保持 NaN
```

### 5. 校验

```python
report = store.validate()       # 检查：空矩阵/inf -> error，NaN 占比超阈值 -> warn
print(report)                   # 逐类型一行；report.ok 为整体判定
```

## 观测报告（HTML）

人肉看 CSV / 静态 PNG 的替代：一键生成交互式离线单文件 HTML，
双击即看、可分发（全部 JS 内联，无外部引用）。需要可选依赖
`pip install plotly`（或 `pip install "profmw[viz]"`，不影响中间件本体）。

```bash
python -m profmw.report run_dummy/ -o report.html      # 命令行
# 可选: --types ddr_bw --prefix CH --top 10
```

```python
from profmw.report import export_report
export_report(store, "report.html", types=["dram_bw"], t_range=(0.0, 10000.0))
```

每个 profiling 类型一个区块，含四种视图（复用 `window_series` 切片逻辑，
支持同样的 modules/prefix/t_range 过滤）：

| 视图 | 用途 |
|---|---|
| 热力图（模块×时间） | 一眼看出通道混叠、热点、空洞等分布问题 |
| 波形叠加图 | 每模块一条曲线；默认亮出均值 Top-N，点图例切换；框选缩放、悬停读数 |
| 模块排名图 | 按窗口均值降序，前 Top-N 标红，快速锁定异常模块 |
| 统计表 | 每模块 窗口数/均值/最大/最小/方差/NaN 占比，与 `aggregate` 口径一致 |

## Dummy 数据（测试用）

```python
from profmw.dummy import make_dummy_store, make_dummy_records

store = make_dummy_store(aliasing=True, hotspot=True)
# ddr_bw:              96 个 DDR 通道（CH00~CH95）
#   aliasing=True      注入"单通道混叠"病态分布（少数通道 ~99，其余 ~1）
# core_read_outstanding: 120 个核（CORE000~CORE119）
#   hotspot=True       注入热点核
# seed 参数保证确定性；make_dummy_records() 返回 raw 长表供测试接入流程
```

命令行生成数据集（`run_dummy/` 中间件目录 + `run_dummy_raw/` 矩阵 CSV）：

```bash
python examples/make_dummy.py --aliasing
python examples/demo.py          # 端到端演示：构建三组矩阵(ddr_bw/96通道rx_req深度/
                                 #   120核outstanding)->保存->读取->窗口统计->HTML报告
```

## API 速查

| 接口 | 说明 |
|---|---|
| `add_matrix(ptype, df, meta=)` | 矩阵形态接入（检查去重、归一 float 列标、排序） |
| `add_type_from_records(ptype, rec, window=, bucket_how=, meta=)` | 长表分桶转矩阵 |
| `add_type_from_histogram(ptype, rec, bins=, quantiles=, meta=)` | 硬件直方图计数 → 分位数矩阵 |
| `save(path, zip_pack=False)` / `load(path)` | CSV+meta.json 目录（或 zip）读写 |
| `list_types()` / `get_matrix(ptype)` | 类型清单（含 meta）/ 矩阵副本 |
| `aggregate(ptype, modules=, module_prefix=, t_range=, agg=, skipna=, interp=)` | 沿时间轴聚合观测值 |
| `window_series(...)` | 逐窗口切片（同参数） |
| `validate(nan_ratio_warn=)` | 数据健康检查 |
| `profmw.report.export_report(store, out, ...)` | HTML 观测报告（可选依赖 plotly） |

## 设计约定

- 各类型时间轴独立，不做重采样对齐；跨类型联合分析由上层显式处理
- 模块名平铺，组聚合靠 `modules` 列表或 `module_prefix` 前缀，不做层级解析
- 统计口径（sum/mean/...）查询时现算，存储只保存窗口基础统计值
- 单 run 单目录；多 run 对比属上层（L1 期望层）逻辑

## 测试

```bash
cd bot_ir && python -m pytest tests -v      # 59 个用例
```
