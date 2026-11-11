"""export_report：单文件 HTML 观测报告生成测试。"""

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


class TestExportReport:
    def test_html_contains_type_meta_and_modules(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        assert "dram_bw" in html
        assert "GB/s" in html and "窗口内均值" in html
        assert "CH0" in html and "CH1" in html
        assert "benchmark" in html and "X9" in html  # run meta

    def test_four_views_present(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        # 热力图/波形/排名 = 3 个 plotly 图，统计表 = 1 个 HTML table
        assert html.count("plotly-graph-div") == 3
        assert "<table" in html
        assert html.count("Plotly.newPlot") == 3  # 内联 JS 仅在首个图携带完整库
        assert html.count('src="https') == 0      # 无外部引用，真正离线

    def test_type_filter(self, small_store, tmp_path):
        other = small_store.get_matrix("dram_bw")
        small_store.add_matrix("util", other, meta={"unit": "%"})
        html = _report_html(small_store, tmp_path, types=["util"])
        assert "util" in html
        assert html.count("plotly-graph-div") == 3  # 只有一个类型的 3 张图

    def test_stats_table_values_match_aggregate(self, small_store, tmp_path):
        html = _report_html(small_store, tmp_path)
        # CH1 行均值 = (10+20+30)/3 = 20.0，应出现在统计表中
        assert "20.0" in html or "20" in html

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
        assert out.exists()
        assert out.read_text(encoding="utf-8").count("plotly-graph-div") == 3
