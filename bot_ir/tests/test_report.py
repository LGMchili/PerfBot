"""export_report：多面板 HTML 观测器测试（时间轴联动 + 选框放大区间统计）。"""

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
        assert "panels.length > 1" in app       # 按数量而非下标控制删除入口
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
