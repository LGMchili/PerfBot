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
- **类型 meta**：模块分组（module）、单位（unit）、统计口径（stat）、描述、来源

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

**分维聚合** `magg`/`tagg`（与 `agg` 互斥）：magg 沿模块轴、tagg 沿时间轴，
固定先模块后时间；省略某维即保留该维（返回 Series）：

```python
# 逐窗口跨模块求和 -> 对时间取平均 -> 单值（如"平均每窗口 ring 总失败数"）
store.aggregate("noc_UpOutFail", module_prefix="R01_",
                magg="sum", tagg="mean")
store.aggregate("noc_UpOutFail", module_prefix="R01_", magg="sum")
# -> Series（标签=时间）：Σ 曲线的 API 形态
store.aggregate("noc_left_util", tagg="mean")
# -> Series（标签=模块）：每节点的窗口均值，统计表口径
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

## NoC ring 数据

层级数据（ring → 通道 → CS 节点 → 指标 × 时间）的接入与观测：

**数据约定**：指标即类型（每指标一个矩阵），行标 = `{R}{ring:02d}_{通道}_{CS}{节点:02d}`
（通道 ∈ `Req / Snp / Rsp / Dat`，如 `R03_Rsp_CS13`）；ring/CS 零填充保证字典序 =
物理序。**前缀过滤即层级聚合**：`module_prefix="R01_"` = 整条 ring，
`"R01_Rsp_"` = 单通道。层级声明写入 run meta（`levels`，入库时自动）。

**接入**：raw 目录 = `manifest.json`（各指标 unit/stat meta）+ 每指标一张矩阵 CSV：

```python
from profmw import ProfilingStore
from profmw.noc import add_noc_raw

store = ProfilingStore(meta={"benchmark": "...", "chip": "..."})
add_noc_raw(store, "noc_raw/")        # 校验行标符合层级约定，levels 写入 run meta
```

**ring 拓扑框**（并入主观测器，见下节「观测器」）：页面顶部「＋ 新增拓扑框」
可随时添加一个 ring 流量窗口——每条 ring 一条横向 CS 方框带，方框四角 =
`左上 UpInFail｜右上 UpOutFail｜左下 DownInFail｜右下 DownOutFail`（数值+按各指标
全局色标染色）；相邻方框之间与首尾两端各一组箭头：`→` = 左侧 CS 的 right_util、
`←` = 右侧 CS 的 left_util，颜色=强度、悬停看数值；ring 首尾环绕。框内自带
通道下拉、ring 勾选（含「全选」，默认全选，部分隐藏时显示半选）、**开始/结束
时间输入框**（按最近采样点落位，起止反填自动纠正，与拖拽条双向同步）、
拖拽时间条（选点/选窗口，窗口时显示均值）、回放，以及拓扑图的**缩放（−/＋）
与「还原」**（0.5×~3×，还原回 100%）。

指标 = 4 个失败计数（`noc_UpInFail`/`noc_UpOutFail`/`noc_DownInFail`/`noc_DownOutFail`）
+ 2 个链路占用率（`noc_left_util`/`noc_right_util`）。缺省自动收录行标符合
`R_C_CS` 三段约定的类型，显式指定不符合约定的类型会报错；`--max-points` 限制每类型
采样点（模块数×窗口数，默认 300000），超限时**仅压缩时间轴**——count 类相邻窗口
求和、其余取均值，模块/CS 完整保留（用于 `--ring` 独立报告）。

```bash
python examples/demo.py               # NoC 接入已并入 demo：生成 8 ring × 4 通道 × 25 CS
                                      # 数据（R03_Rsp 通道 CS12~14 注入环段拥塞），
                                      # 所有产物（含 raw 契约示例 raw/）统一落在 run_demo/
python -m profmw.report run_demo -o run_demo_report.html   # 观测器（波形/拓扑/时延框）
python -m profmw.report run_demo --ring -o ring.html       # 可选的独立行列式报告
```

## 时延分段统计（AIC 路径）

每个 AIC 一份「时延路径」矩阵：行 = 路径（如 `AIC→Ring`、`Ring→HBMC`），
列 = 时间窗口，值 = 该路径在窗口内的**平均时延**（已按段内请求平均）。约定 meta：

```python
store.add_matrix("lat_AIC00", df, meta={
    "module": "AIC",              # 第一级下拉：模块（如 AIC）
    "label": "AIC0",              # 第二级下拉的短名
    "unit": "ns",
    "kind": "latency_path",       # 观测器据此识别并只给该类型「时延堆叠」视图
    "stat": "窗口内平均", "description": "AIC0 各时延路径平均时延"})
```

观测器顶部「＋ 新增时延框」添加面板，两级下拉：**先选模块（如 `AIC`），再选
`AIC0`/`AIC1`/…/`nanmean`**，默认 `nanmean`——该模块组内全部类型逐元素取均值
（忽略 NaN）。只有「时延堆叠」视图，各路径平均时延**绝对堆叠、不做归一化**，
堆叠顶边 = 端到端平均时延（四条路径 20/50/50/20 ns 则顶边 = 140 ns）；时间轴
与其它面板联动。**波形框与时延框内容隔离、互不可选**：波形框的模块/类型下拉
不出现时延类型，时延框只列 `latency_path` 类型且只有时延堆叠视图。demo 生成
8 个 AIC（`lat_AIC00`~`lat_AIC07`，短名 `AIC0`~`AIC7`）× 4 条路径，`AIC03`
的 `Ring→HBMC` 在 20~30 号窗口抬升作为异常示例。

## 观测器（HTML）

人肉看 CSV / 静态 PNG 的替代：多面板交互观测器，离线单文件 HTML，
双击即看、可分发（全部 JS 内联，无外部引用）。需要可选依赖
`pip install plotly`（或 `pip install "profmw[viz]"`，不影响中间件本体）。
波形框、ring 拓扑框与 AIC 时延框在**同一页面**：NoC 数据存在时出现
「＋ 新增拓扑框」，时延类型存在时出现「＋ 新增时延框」，否则按钮自动隐藏。
三类内容互不串选：波形框不列 NoC（`R_C_CS` 行标）与时延类型，NoC 只由拓扑框
展示、时延只由时延框展示。`--ring` 仍可生成纯 DOM、无需 plotly 的独立页面
（拓扑图/矩阵表）。

```bash
python -m profmw.report run_dummy/ -o report.html      # 命令行
# 可选: --types ddr_bw --prefix CH --top 10
python -m profmw.report run_demo/ --ring -o ring.html  # 可选：独立 ring 页面
# 可选: --max-points（默认 300000，超限时仅压缩时间轴）
```

```python
from profmw.report import export_report
export_report(store, "report.html", types=["dram_bw"], t_range=(0.0, 10000.0))
```

界面：默认一个面板，点击 **“＋ 新增波形框”** 追加面板做同轴对比；点面板右上角
**×** 移除面板（任意面板均可删，至少保留一个）。每个面板通过 **模块 → 统计项
两级下拉**选择数据类型（分组来自类型 meta 的 `module` 字段，缺省归"默认"组），
再选视图（波形 / 热力图 / 统计表），面板下方是统计条。

- **框选测量（不缩放）**：拖拽框选 = 测量区间，波形原地不动，统计条实时给出
  选区内的 均值/最大/最小/样本数（波形视图只统计图例可见的曲线）；未选中时
  显示全区间基线，双击清除选区；框选只取水平（时间）范围
- **缩放**：工具栏按钮（滚轮缩放已关闭），双击复位；**时间轴联动**——任一面板
  缩放/复位，其余波形、热力图面板同步，便于"带宽 vs 队列深度"同轴对比
- **Σ 求和曲线**：波形视图面板可勾选，追加一条红色加粗曲线 = 当前图例可见
  模块的逐时间点求和；点图例选择参与求和的模块，曲线实时重算；显示该曲线时，
  框选统计会单独追加一段 Σ 曲线的选区内 均值/最大/最小/样本数
- 波形视图默认亮出窗口均值 Top-N，点图例切换显隐，图例开关也会刷新统计口径
- 热力图（模块×时间）一眼看出通道混叠、热点、空洞等分布问题
- 统计表视图给出每模块 窗口数/均值/最大/最小/方差/NaN 占比全量口径

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
python examples/demo.py          # 端到端演示：构建矩阵数据集 + NoC raw 接入 ->
                                 #   保存 -> 读取 -> 窗口/层级统计 -> HTML 报告
```

## API 速查

| 接口 | 说明 |
|---|---|
| `add_matrix(ptype, df, meta=)` | 矩阵形态接入（检查去重、归一 float 列标、排序） |
| `add_type_from_records(ptype, rec, window=, bucket_how=, meta=)` | 长表分桶转矩阵 |
| `save(path, zip_pack=False)` / `load(path)` | CSV+meta.json 目录（或 zip）读写 |
| `list_types()` / `get_matrix(ptype)` | 类型清单（含 meta）/ 矩阵副本 |
| `aggregate(ptype, modules=, module_prefix=, t_range=, agg=, magg=, tagg=, skipna=, interp=)` | 聚合观测值：agg=池化单值；magg/tagg=分维（Series 或单值） |
| `window_series(...)` | 逐窗口切片（同参数） |
| `validate(nan_ratio_warn=)` | 数据健康检查 |
| `profmw.report.export_report(store, out, ...)` | 观测报告：波形/热力图/统计表 + 可新增 ring 拓扑框 / AIC 时延框（可选依赖 plotly） |
| `profmw.report.export_ring_report(store, out, ...)` | 独立 ring 页面：拓扑图/矩阵表（纯 DOM，无 plotly） |

## 设计约定

- 各类型时间轴独立，不做重采样对齐；跨类型联合分析由上层显式处理
- 模块名平铺，组聚合靠 `modules` 列表或 `module_prefix` 前缀，不做层级解析
- 统计口径（sum/mean/...）查询时现算，存储只保存窗口基础统计值
- 单 run 单目录；多 run 对比属上层（L1 期望层）逻辑

## 测试

```bash
cd bot_ir && python -m pytest tests -v      # 59 个用例
```
