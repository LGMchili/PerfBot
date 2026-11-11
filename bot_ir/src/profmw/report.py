"""单文件 HTML 观测报告：热力图 / 波形叠加 / 模块排名 / 统计表。

依赖 plotly（可选：pip install "profmw[viz]"）。生成的 HTML 内联全部 JS，
离线双击即开，可直接分发。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from profmw.store import ProfilingStore

_CSS = """
body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 24px auto;
       max-width: 1200px; color: #222; }
h1 { font-size: 22px; } h2 { font-size: 18px; margin-top: 40px;
       border-left: 4px solid #4a90d9; padding-left: 10px; }
.meta { color: #666; font-size: 14px; }
table { border-collapse: collapse; font-size: 13px; margin-top: 12px; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f5f7fa; }
td:first-child { font-weight: 500; }
"""


def _fig_html(fig, first: bool) -> str:
    return fig.to_html(
        full_html=False,
        include_plotlyjs="inline" if first else False,
        config={"responsive": True},
    )


def _heatmap(df: pd.DataFrame, title: str):
    import plotly.graph_objects as go

    fig = go.Figure(go.Heatmap(
        x=list(df.columns), y=list(df.index), z=df.to_numpy(dtype=float),
        colorbar={"title": {"text": "值"}},
        hovertemplate="%{y}<br>t=%{x}<br>值=%{z}<extra></extra>",
    ))
    fig.update_layout(title=title, xaxis_title="时间", yaxis_title="模块")
    return fig


def _lines(df: pd.DataFrame, title: str, top_n: int):
    import plotly.graph_objects as go

    order = df.mean(axis=1).sort_values(ascending=False).index
    fig = go.Figure()
    for name in order:
        fig.add_trace(go.Scatter(
            x=list(df.columns), y=df.loc[name].to_numpy(dtype=float),
            name=str(name), mode="lines",
            visible=(True if order.get_loc(name) < top_n else "legendonly"),
        ))
    fig.update_layout(title=title, xaxis_title="时间", yaxis_title="值",
                      hovermode="x unified", legend={"itemsizing": "trace"})
    return fig


def _ranking(df: pd.DataFrame, title: str, top_n: int):
    import plotly.graph_objects as go

    means = df.mean(axis=1).sort_values(ascending=False)
    colors = ["#d62728" if i < top_n else "#4a90d9" for i in range(len(means))]
    fig = go.Figure(go.Bar(
        x=means.to_numpy(), y=[str(i) for i in means.index],
        orientation="h", marker_color=colors,
        hovertemplate="%{y}<br>窗口均值=%{x}<extra></extra>",
    ))
    fig.update_layout(title=title, height=max(300, 18 * len(means)),
                      xaxis_title="窗口均值", yaxis_title="模块")
    fig.update_yaxes(autorange="reversed")
    return fig


def _stats_table(df: pd.DataFrame) -> str:
    stats = pd.DataFrame({
        "窗口数": df.notna().sum(axis=1),
        "均值": df.mean(axis=1),
        "最大": df.max(axis=1),
        "最小": df.min(axis=1),
        "方差": df.var(axis=1),
        "NaN占比": df.isna().mean(axis=1),
    }).sort_values("均值", ascending=False)
    head = "".join(f"<th>{c}</th>" for c in ["模块", *stats.columns])
    rows = []
    for name, row in stats.iterrows():
        cells = "".join(
            f"<td>{row[c]:.4%}</td>" if c == "NaN占比" else f"<td>{row[c]:.4g}</td>"
            for c in stats.columns
        )
        rows.append(f"<tr><td>{name}</td>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def export_report(
    store: ProfilingStore,
    out,
    types: list[str] | None = None,
    modules: list[str] | None = None,
    module_prefix: str | None = None,
    t_range=None,
    top_n: int = 20,
) -> Path:
    """生成单文件 HTML 观测报告，返回文件路径。"""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        raise ImportError(
            "生成报告需要 plotly：pip install plotly（或 pip install 'profmw[viz]'）"
        ) from None

    selected = types if types is not None else list(store.list_types())
    if not selected:
        raise ValueError("没有任何 profiling 类型")

    meta_line = " · ".join(f"{k}={v}" for k, v in store.meta.items()) or "（无 run meta）"
    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        f"<title>profmw 报告</title><style>{_CSS}</style></head><body>",
        "<h1>profiling 观测报告</h1>", f"<p class='meta'>{meta_line}</p>",
    ]
    fig_no = 0
    for ptype in selected:
        tmeta = store.list_types().get(ptype, {})
        df = store.window_series(ptype, modules=modules, module_prefix=module_prefix,
                                 t_range=t_range)
        parts.append(f"<h2>{ptype}</h2>")
        desc = " · ".join(f"{k}={v}" for k, v in tmeta.items())
        if desc:
            parts.append(f"<p class='meta'>{desc}</p>")
        parts.append(f"<p class='meta'>{df.shape[0]} 模块 × {df.shape[1]} 窗口</p>")
        figs = [
            _heatmap(df, "模块×时间 热力图"),
            _lines(df, f"波形叠加图（默认显示窗口均值 Top{top_n}，点击图例切换）", top_n),
            _ranking(df, f"模块排名（窗口均值降序，前 {top_n} 标红）", top_n),
        ]
        for fig in figs:
            parts.append(_fig_html(fig, first=(fig_no == 0)))
            fig_no += 1
        parts.append("<h3>统计表（按均值降序）</h3>")
        parts.append(_stats_table(df))
    parts.append("</body></html>")

    out = Path(out)
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m profmw.report", description="从 run 目录生成 HTML 观测报告")
    parser.add_argument("run", help="run 目录（或 .zip）路径")
    parser.add_argument("-o", "--out", default="report.html", help="输出 HTML 路径")
    parser.add_argument("--types", nargs="*", help="只报告指定类型")
    parser.add_argument("--prefix", help="模块名前缀过滤")
    parser.add_argument("--top", type=int, default=20, help="波形默认显示的 Top-N 模块数")
    args = parser.parse_args(argv)
    store = ProfilingStore.load(args.run)
    out = export_report(store, args.out, types=args.types, module_prefix=args.prefix,
                        top_n=args.top)
    print(f"报告已生成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
