"""export_report / export_ring_report 观测器测试。"""

import numpy as np
import pandas as pd
import pytest

from profmw import ProfilingStore


@pytest.fixture
def small_store():
    df = pd.DataFrame(
        [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
        index=["CH0", "CH1"],
        columns=[100.0, 200.0, 300.0],
    )
    store = ProfilingStore(meta={"benchmark": "b", "chip": "X9"})
    store.add_matrix("dram_bw", df, meta={"unit": "GB/s", "stat": "窗口内均值"})
    return store


def _report_html(store, tmp_path, **kw):
    pytest.importorskip("plotly")
    from profmw.report import export_report
    out = tmp_path / "r.html"
    export_report(store, out, **kw)
    assert out.exists()
    return out.read_text(encoding="utf-8")


def _app_js(html):
    """应用脚本 = 最后一个 <script> 块（前面的块是内联 plotly 库，
    其自身含 plotly_selected 等事件名字符串，不能作为断言对象）。"""
    return html.rsplit("<script>", 1)[1]


class TestExportReport:
    def test_multi_panel_layout_with_add_button(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        assert html.count('id="panels"') == 1        # 面板容器
        assert "新增波形框" in html                   # 追加面板入口
        for view in ("波形", "热力图", "统计表"):  # 每面板可选视图
            assert view in html
        assert "排名" not in html                    # 排名视图已移除

    def test_type_dropdown_sorted_by_name(self, small_store, tmp_path):
        # 类型下拉按名称排序显示，而非入库顺序
        app = _app_js(_report_html(small_store, tmp_path))
        assert "Object.keys(DATA).sort()" in app

    def test_cascading_type_dropdown(self, small_store, tmp_path):
        # 类型选择两级化：模块 -> 统计项；分组来自类型 meta 的 module 字段
        app = _app_js(_report_html(small_store, tmp_path))
        assert 'class="p-mod"' in app and 'class="p-stat"' in app  # 两个下拉
        assert "moduleOf" in app                          # 按 meta.module 分组
        assert "默认" in app                               # 无 module meta 归入默认组

    def test_header_shows_only_run_meta(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        head = html.split("<body>")[1].split("<script>")[0]
        assert "profiling 观测" not in head          # 开头不再有大标题
        assert "benchmark=b" in head and "chip=X9" in head  # 仅显示 run 信息

    def test_data_embedded_as_json_with_meta(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        assert "const DATA" in html and '"dram_bw"' in html
        assert '"CH0"' in html and '"CH1"' in html
        assert "GB/s" in html and "benchmark" in html and "X9" in html

    def test_selection_stats_and_axis_sync_js(self, small_store, tmp_path):
        # 框选=测量（不缩放），统计条显示选区/全区间；缩放仍联动时间轴
        app = _app_js(_report_html(small_store, tmp_path))
        assert "plotly_selected" in app     # 框选事件 -> 选区统计
        assert "plotly_deselect" in app     # 双击清除 -> 回到全区间基线
        assert "plotly_relayout" in app     # 缩放事件 -> 时间轴联动
        assert "plotly_restyle" in app      # 图例开关 -> 统计口径刷新
        assert "scrollZoom: false" in app   # 关闭滚轮缩放（缩放走工具栏）
        assert "sameRange" in app          # 联动前数值比较，防 relayout 乒乓挂死
        assert "selectdirection" in app     # 水平框选（只取时间区间）
        assert "均值" in app and "最大" in app and "最小" in app
        assert "xaxis.range" in app         # 时间轴联动载体

    def test_sum_curve_feature(self, small_store, tmp_path):
        # Σ 求和曲线：开关 + 可见模块逐点求和 + 不污染统计口径
        app = _app_js(_report_html(small_store, tmp_path))
        assert "求和曲线" in app           # 面板开关
        assert "Σ 可见模块" in app          # 求和曲线图例名
        assert "sumCurve" in app            # 逐时间点求和（跟随图例可见性）
        assert 'startsWith("Σ")' in app    # 统计/求和口径排除 Σ 曲线自身

    def test_any_panel_removable_but_keep_one(self, small_store, tmp_path):
        # 面板删除：数量 >= 2 时每个面板（含第一个）都有 ×；只剩一个时隐藏 ×
        app = _app_js(_report_html(small_store, tmp_path))
        assert "panels.length + topoPanels.length > 1" in app  # 波形框+拓扑框统一计数
        assert 'i ? "" : "none"' not in app     # 不再按下标永久保留第一个面板

    def test_ranking_view_removed(self, small_store, tmp_path):
        app = _app_js(_report_html(small_store, tmp_path))
        assert '"ranking"' not in app           # 视图选项移除
        assert 'type: "bar"' not in app         # 排名条形图渲染移除
        assert "排名图无需框选" not in app        # 排名视图统计文案移除

    def test_selection_stats_include_sum_curve_when_shown(self, small_store, tmp_path):
        # 显示 Σ 曲线时，选框统计单独追加 Σ 段；不显示时保持原口径
        app = _app_js(_report_html(small_store, tmp_path))
        assert "sumStatsIn" in app               # 选区内 Σ 曲线统计
        assert "Σ: 均值" in app                  # 统计条单独追加 Σ 一段
        assert 'pn.view === "lines" && pn.showSum' in app  # 仅在显示 Σ 时追加

    def test_fixed_chart_heights_prevent_growth(self, small_store, tmp_path):
        # 视图切换会把内联高度写死在图 div 上，重渲染时 autosize 按残留高度
        # 自适应 -> 面板只涨不缩。修复：显式固定高度 + 重渲染前清残留。
        app = _app_js(_report_html(small_store, tmp_path))
        assert "layout.height = 460" in app        # 波形/热力图固定高度
        assert 'chartEl.style.height = ""' in app  # 重渲染前清除内联高度残留

    def test_default_panel_sum_curve(self, tmp_path):
        # 首屏 SMMU.TotalBw 默认打开 Σ 求和曲线，其他面板默认关闭
        store = ProfilingStore()
        store.add_matrix("SMMU.TotalBw",
                         pd.DataFrame([[1.0, 2.0]], index=["SMMU"],
                                      columns=[0.0, 1.0]),
                         meta={"module": "SMMU", "unit": "GB/s"})
        store.add_matrix("dram_bw",
                         pd.DataFrame([[1.0, 2.0]], index=["CH0"],
                                      columns=[0.0, 1.0]),
                         meta={"unit": "GB/s"})
        app = _app_js(_report_html(store, tmp_path))
        assert "sumOn = false" in app               # 新增面板默认关闭
        assert "initType === DEFAULT_WAVE" in app   # 首屏默认打开
        assert "sumCb.checked = sumOn" in app

    def test_ring_matrix_view(self, small_store, tmp_path):
        # ring矩阵 仍不在波形视图里；ring 观测以「新增拓扑框」提供
        html = _report_html(small_store, tmp_path)
        app = _app_js(html)
        assert '"ring"' not in app                # 视图选项未新增 ring
        assert "ring矩阵" not in app               # 矩阵表仅在独立 --ring 报告中
        assert "新增拓扑框" in html                 # 波形页的 ring 观测入口
        assert "addTopoPanel" in app

    def test_latency_panel_stacks_absolute_latency(self, tmp_path):
        # 时延框：各路径平均时延绝对堆叠（不归一化），顶边=端到端平均时延
        store = ProfilingStore()
        df = pd.DataFrame([[20.0, 21.0], [50.0, 55.0]],
                          index=["AIC→Ring", "Ring→HBMC"], columns=[0.0, 100.0])
        store.add_matrix("lat_AIC00", df,
                         meta={"module": "latency", "unit": "ns",
                               "kind": "latency_path"})
        html = _report_html(store, tmp_path)
        app = _app_js(html)
        assert "新增时延框" in html                    # 入口
        assert "LATENCY_TYPES" in app and "latency_path" in app
        assert "stackgroup" in app                     # 绝对时延堆叠
        assert "groupnorm" not in app                  # 不做占比归一化
        assert "总时延" in app                          # 统计条按总和口径
        assert "viewsFor" in app                       # 仅时延类型多出堆叠视图
        assert '"nanmean"' in app                      # 下拉项（不带"多模块"前缀）
        assert "LAT_MEAN_FOR" in app                   # 每个 (模块,统计) 一份 nanmean
        assert "meta.label" in app                     # 统计项用短标签（AIC0…）
        assert "p-latstat" in app and "lat_stat" in app  # 时延统计二级下拉
        assert "SMMU.TotalBw" in app                   # 首屏优先类型

    def test_latency_stat_dropdown_lists_rd_wr(self, tmp_path):
        # 同一 AIC 的不同时延统计（RdLat/WrLat）作为独立二级选项
        store = ProfilingStore()
        df = pd.DataFrame([[20.0], [50.0]],
                          index=["AIC→Ring", "Ring→HBMC"], columns=[0.0])
        for stat in ("RdLat", "WrLat"):
            store.add_matrix(f"lat_AIC0.{stat}", df,
                             meta={"module": "AIC", "label": "AIC0",
                                   "lat_stat": stat, "unit": "ns",
                                   "kind": "latency_path"})
        app = _app_js(_report_html(store, tmp_path))
        assert "RdLat" in app and "WrLat" in app
        assert "latStatOf" in app

    def test_waveform_panels_exclude_noc_types(self, tmp_path):
        # NoC（R_C_CS 行标）只由拓扑框展示，不再出现在波形框类型列表
        store = _noc_store()
        store.add_matrix("dram_bw",
                         pd.DataFrame([[1.0]], index=["CH0"], columns=[0.0]),
                         meta={"unit": "GB/s"})
        app = _app_js(_report_html(store, tmp_path))
        assert "!isLatency(t) && !ringShaped(DATA[t])" in app  # 波形框过滤
        assert "!ringShaped(DATA[t])" in app

    def test_wave_and_latency_panels_isolated(self, tmp_path):
        # 波形框只列普通类型，时延框只列 latency_path 类型，互不可选
        store = ProfilingStore()
        store.add_matrix("dram_bw",
                         pd.DataFrame([[1.0]], index=["CH0"], columns=[0.0]),
                         meta={"unit": "GB/s"})
        store.add_matrix("lat_AIC00",
                         pd.DataFrame([[20.0]], index=["AIC→Ring"], columns=[0.0]),
                         meta={"module": "latency", "unit": "ns",
                               "kind": "latency_path"})
        app = _app_js(_report_html(store, tmp_path))
        assert "isLatency" in app
        assert "WAVE_TYPES = ALL_TYPES.filter" in app
        assert "kindOk = t => panel.latency" in app

    def test_topo_panel_embeds_ring_data_and_ip_names(self, tmp_path):
        # 拓扑框复用同一份 DATA：六指标四角/箭头 + 槽位 IP 名
        store = _noc_store()
        store.meta["noc_node_names"] = {"R00_CS00": "HBMC", "R00_CS01": "MATA"}
        store.meta["noc_ring_die"] = {"R00": "die0", "R01": "die1"}
        html = _report_html(store, tmp_path)
        app = _app_js(html)
        assert "die1" in app                            # ring->die 映射内嵌
        assert "RING_METRICS" in app and "RING_TYPES" in app
        assert "NODE_NAMES" in app and "HBMC" in app    # IP 名内嵌
        assert 'topoBtn.style.display = "none"' in app  # 无 NoC 数据时自动隐藏入口
        assert "p-die" in app and "RING_DIE" in app      # 按 die 下拉筛选
        assert 'class="p-t0"' in app                    # 开始时间输入框
        assert 'class="p-t1"' in app                    # 结束时间输入框
        assert "nearestIdx" in app                      # 时间输入按采样点落位
        assert "p-zin" in app and "p-zreset" in app      # 缩放 / 还原
        head = html.split("<body>")[1].split("<script>")[0]
        assert "HBMC" not in head                       # 页头不糊 IP 名映射
        assert "die1" not in head                       # 页头不糊 ring->die 映射


def _noc_store():
    df = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["R00_Req_CS00", "R00_Req_CS01"],
        columns=[100.0, 200.0],
    )
    store = ProfilingStore(meta={"levels": [{"key": "R", "width": 2},
                                            {"key": "C", "width": 2},
                                            {"key": "CS", "width": 2}]})
    store.add_matrix("noc_up_fail_count", df,
                     meta={"module": "dummy", "unit": "count"})
    return store


class TestExportRingReport:
    def _ring_html(self, store, tmp_path, **kw):
        from profmw.report import export_ring_report
        out = tmp_path / "ring.html"
        export_ring_report(store, out, **kw)
        return out.read_text(encoding="utf-8")

    def test_ring_page_has_table_timebar_channel(self, tmp_path):
        html = self._ring_html(_noc_store(), tmp_path)
        assert "n-track" in html and "n-h1" in html and "n-h2" in html  # 双柄时间条
        assert "回放" in html and "通道" in html
        assert 'id="ch-sel"' not in html           # 通道不再单独下拉
        assert "${o.type}.${o.channel}" in html    # 统计项按「类型.通道」平铺
        assert "新增波形框" not in html            # 非面板体系
        assert "plotly" not in html.lower()        # 纯 DOM，无 plotly 依赖

    def test_waveform_types_excluded_by_auto_selection(self, tmp_path):
        store = _noc_store()
        store.add_matrix("ddr_bw", pd.DataFrame([[1.0]], index=["CH0"],
                                                columns=[1.0]))
        html = self._ring_html(store, tmp_path)
        assert '"noc_up_fail_count"' in html
        assert '"ddr_bw"' not in html              # 非 R_C_CS 行标类型不进 ring 报告

    def test_explicit_non_noc_type_rejected(self, tmp_path):
        store = _noc_store()
        store.add_matrix("ddr_bw", pd.DataFrame([[1.0]], index=["CH0"],
                                                columns=[1.0]))
        from profmw.report import export_ring_report
        with pytest.raises(ValueError, match="行标"):
            export_ring_report(store, tmp_path / "r.html", types=["ddr_bw"])

    def _embedded_data(self, html):
        import json
        blob = html.split("const DATA = ", 1)[1]
        return json.loads(blob.split("};", 1)[0] + "}")

    def test_max_points_compresses_time_not_modules(self, tmp_path):
        store = ProfilingStore()
        store.add_matrix("noc_util",
                         pd.DataFrame(np.arange(8 * 100, dtype=float).reshape(8, 100),
                                      index=[f"R{i:02d}_Req_CS{i:02d}" for i in range(8)],
                                      columns=np.arange(100, dtype=float) * 10),
                         meta={"unit": "%"})
        html = self._ring_html(store, tmp_path, max_points=200)
        d = self._embedded_data(html)["noc_util"]
        assert len(d["modules"]) == 8               # 模块不动
        assert len(d["times"]) == 25                # 8×100=800 > 200 -> k=4 -> 100/4
        assert d["downsample"] == 4
        # mean 合并：首模块首 4 个值 (0,1,2,3) 的均值 = 1.5
        assert abs(d["values"][0][0] - 1.5) < 1e-9

    def test_max_points_count_unit_merges_by_sum(self, tmp_path):
        store = ProfilingStore()
        store.add_matrix("noc_cnt",
                         pd.DataFrame(np.ones((4, 100)),
                                      index=[f"R{i:02d}_Req_CS{i:02d}" for i in range(4)],
                                      columns=np.arange(100, dtype=float)),
                         meta={"unit": "count"})
        html = self._ring_html(store, tmp_path, max_points=200)  # 4×100=400 -> k=2
        d = self._embedded_data(html)["noc_cnt"]
        assert d["downsample"] == 2
        assert abs(d["values"][0][0] - 2.0) < 1e-9  # count -> 相邻窗口求和
        assert d["dsh"] == "sum"

    def test_no_downsample_when_under_limit(self, tmp_path):
        html = self._ring_html(_noc_store(), tmp_path, max_points=10000)
        d = self._embedded_data(html)["noc_up_fail_count"]
        assert d["downsample"] is None
        assert len(d["times"]) == 2

    def test_topo_view(self, tmp_path):
        # 拓扑图模式：CS 方框四角 = 四个失败指标，箭头 = 链路占用率，首尾环绕
        html = self._ring_html(_noc_store(), tmp_path)
        assert "拓扑图" in html and "矩阵表" in html       # 模式切换
        assert "topo-ring" in html and "tbox" in html      # 拓扑 DOM 结构
        assert "buildTopo" in html and "updateTopo" in html
        for metric in ("noc_UpInFail", "noc_UpOutFail",    # 四角指标接线
                       "noc_DownInFail", "noc_DownOutFail",
                       "noc_left_util", "noc_right_util"):
            assert metric in html

    def test_topo_arrow_values_and_node_names(self, tmp_path):
        # 箭头标注利用率数值；方框显示硬件 IP 名（HBMC/MATA/SMMU...）
        html = self._ring_html(_noc_store(), tmp_path)
        assert "NODE_NAMES" in html                 # IP 名映射内嵌
        assert "fmtShort" in html                   # 箭头/角标紧凑数值
        assert 'class="tip"' in html                # 方框内 IP 名行
        assert 'class="av"' in html                 # 箭头数值 span

    def test_ring_filter_default_all(self, tmp_path):
        # 每条 ring 一个复选框，默认全选（hidden 集合为空），拓扑/矩阵共用过滤
        html = self._ring_html(_noc_store(), tmp_path)
        assert "id='ring-filter'" in html
        assert "hidden: new Set()" in html          # 默认不隐藏任何 ring
        assert "shownRings" in html                 # 两个视图都走同一过滤

    def test_default_max_points_is_300k(self, tmp_path):
        # 不传 max_points 时默认上限 300000：600×501=300600 > 300000 -> k=2
        store = ProfilingStore()
        store.add_matrix("noc_big",
                         pd.DataFrame(np.zeros((600, 501)),
                                      index=[f"R{i:03d}_Req_CS00" for i in range(600)],
                                      columns=np.arange(501, dtype=float)),
                         meta={"unit": "%"})
        d = self._embedded_data(self._ring_html(store, tmp_path))["noc_big"]
        assert d["downsample"] == 2
        assert len(d["times"]) == 251

    def test_fully_offline(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        assert 'src="https' not in html
        assert "Plotly" in html

    def test_type_filter(self, small_store, tmp_path):
        other = small_store.get_matrix("dram_bw")
        small_store.add_matrix("util", other, meta={"unit": "%"})
        html = _report_html(small_store, tmp_path, types=["util"])
        assert '"util"' in html
        assert '"dram_bw"' not in html            # 未选类型不内嵌

    def test_missing_plotly_raises_hint(self, small_store, tmp_path, monkeypatch):
        import sys
        from profmw import report as report_mod
        monkeypatch.setitem(sys.modules, "plotly", None)  # 阻断导入
        with pytest.raises(ImportError, match="viz"):
            report_mod.export_report(small_store, tmp_path / "x.html")


class TestCli:
    def test_main_writes_report(self, small_store, tmp_path):
        run = tmp_path / "run"
        small_store.save(run)
        pytest.importorskip("plotly")
        from profmw.report import main
        out = tmp_path / "rep.html"
        rc = main([str(run), "-o", str(out)])
        assert rc == 0
        html = out.read_text(encoding="utf-8")
        assert html.count('id="panels"') == 1
        assert '"dram_bw"' in html

    def test_main_ring_report_with_max_points(self, tmp_path):
        run = tmp_path / "run"
        _noc_store().save(run)
        from profmw.report import main
        out = tmp_path / "ring.html"
        rc = main([str(run), "--ring", "--max-points", "4", "-o", str(out)])
        assert rc == 0
        html = out.read_text(encoding="utf-8")
        assert "n-track" in html                       # ring 专用页面
        # 2 模块 × 2 窗口 = 4 点 = 上限，不压缩
        assert '"downsample": null' in html
