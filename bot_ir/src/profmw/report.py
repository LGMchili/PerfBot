"""多面板 HTML 观测器：面板可增删、时间轴联动，框选测量不缩放。

依赖 plotly（可选：pip install "profmw[viz]"）。数据以 JSON 内嵌、plotly JS 内联，
离线双击即开。交互约定：拖拽框选 = 测量区间（不缩放，双击清除），统计条显示
选区内的均值/最大/最小/样本数，未选中时显示全区间基线（波形视图只统计
图例可见的曲线）；波形视图可勾选 Σ 求和曲线（可见模块逐时间点求和，随图例
实时重算）；缩放走工具栏按钮（滚轮缩放已关闭），各面板时间轴联动。
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from profmw.store import ProfilingStore

_CSS = """
body { font-family: -apple-system, "PingFang SC", sans-serif; margin: 16px auto;
       max-width: 1280px; color: #222; }
.meta { color: #666; font-size: 13px; margin-bottom: 12px; }
.controls { margin-bottom: 12px; }
.controls button { border: 1px solid #4a90d9; background: #fff; color: #4a90d9;
                   padding: 6px 16px; cursor: pointer; font-size: 14px;
                   border-radius: 4px; }
.controls button:hover { background: #4a90d9; color: #fff; }
.panel { border: 1px solid #e2e6ea; border-radius: 6px; padding: 10px 12px;
         margin-bottom: 14px; background: #fff; }
.panel-head { display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
              flex-wrap: wrap; }
.panel-head select { font-size: 13px; padding: 3px 6px; }
.p-close { border: none; background: none; color: #999; font-size: 16px;
           cursor: pointer; padding: 0 4px; }
.p-close:hover { color: #d62728; }
.p-info { color: #999; font-size: 12px; }
.p-chart { width: 100%; min-height: 360px; max-height: 820px; overflow-y: auto; }
.p-table { max-height: 640px; overflow: auto; }
table { border-collapse: collapse; font-size: 13px; }
th, td { border: 1px solid #ddd; padding: 4px 10px; text-align: right;
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; font-weight: 500; }
th { background: #f5f7fa; position: sticky; top: 0; }
.p-stats { margin-top: 6px; padding: 6px 10px; background: #f5f7fa;
           border-radius: 4px; font-size: 13px; min-height: 18px; }
.views { display: flex; gap: 4px; margin-bottom: 8px; }
.views button { border: 1px solid #ccc; background: #fff; padding: 5px 14px;
                cursor: pointer; font-size: 14px; border-radius: 4px; }
.views button.active { background: #4a90d9; color: #fff; border-color: #4a90d9; }
.ring-filter { display: flex; align-items: center; gap: 8px; font-size: 12px;
               color: #555; margin-bottom: 6px; }
.ring-filter label { display: flex; align-items: center; gap: 2px; cursor: pointer; }
.ring-filter input { margin: 0; }
.ring-filter select { font-size: 12px; padding: 2px 4px; }
.time-inputs { display: flex; align-items: center; gap: 6px; font-size: 12px;
               color: #555; margin-bottom: 6px; }
.time-inputs input { width: 90px; font-size: 12px; padding: 2px 4px; }
.p-zoom { display: flex; align-items: center; gap: 4px; font-size: 12px; }
.p-zoom button { border: 1px solid #ccc; background: #fff; border-radius: 4px;
                 padding: 1px 6px; cursor: pointer; font-size: 12px; }
.p-zoom button:hover { border-color: #4a90d9; color: #4a90d9; }
.p-zoom .p-zval { min-width: 38px; text-align: center; color: #555; }
.p-topo-inner { transform-origin: top left; }
#topo { margin-bottom: 10px; }
.topo-legend { color: #666; font-size: 12px; margin: 4px 0 10px; }
.topo-ring { display: flex; align-items: center; margin-bottom: 6px; }
.topo-name { color: #4a90d9; font-weight: 600; font-size: 12px; margin-right: 6px;
             min-width: 32px; }
.topo-strip { display: flex; align-items: center; }
.tbox { border: 1px solid #c5ccd4; border-radius: 3px; padding: 2px 3px;
        background: #fff; text-align: center; }
.tcs { font-size: 9px; line-height: 10px; color: #666; }
.tip { font-size: 9px; line-height: 10px; color: #4a90d9; font-weight: 600;
       margin-bottom: 1px; min-height: 10px; }
.tgrid { display: grid; grid-template-columns: auto auto; gap: 1px 2px; }
.tcell { font-size: 10px; line-height: 12px; padding: 0 2px; border-radius: 2px;
         min-width: 24px; }
.lnk { display: flex; flex-direction: column; justify-content: center;
       padding: 0 1px; }
.arr { display: flex; align-items: center; font-size: 9px; line-height: 11px;
       padding: 0 1px; border-radius: 2px; color: #333; }
.av { min-width: 22px; text-align: left; }
.p-topo { overflow-x: auto; }
.p-stats b { color: #d62728; }
.p-noc { display: none; align-items: center; gap: 12px; margin-bottom: 6px;
         flex-wrap: wrap; font-size: 13px; }
.n-track { position: relative; flex: 1 1 240px; height: 26px; cursor: pointer;
           touch-action: none; }
.n-track::before { content: ""; position: absolute; left: 0; right: 0; top: 11px;
                   height: 4px; background: #e2e6ea; border-radius: 2px; }
.n-fill { position: absolute; top: 11px; height: 4px; background: #4a90d9;
          border-radius: 2px; }
.n-h { position: absolute; top: 4px; width: 12px; height: 18px; background: #4a90d9;
       border-radius: 3px; cursor: ew-resize; }
.n-time { color: #666; min-width: 90px; }
.n-play { border: 1px solid #ccc; background: #fff; border-radius: 4px;
          padding: 2px 10px; cursor: pointer; font-size: 13px; }
.noc-table td { min-width: 52px; }
"""

_JS_APP = r"""
const DATA = __DATA__;
const TOPN = __TOPN__;
const NODE_NAMES = __NODES__;   // 拓扑框用：CS 槽位 IP 名（如 R00_CS00 -> HBMC）
const RING_DIE = __RING_DIE__;  // ring -> die（如 R00 -> die0），缺省全归 die0
const ALL_TYPES = Object.keys(DATA).sort();
// 模块分组：类型 meta 的 module 字段；缺省归"默认"组
const moduleOf = ptype => (DATA[ptype].meta && DATA[ptype].meta.module) || "默认";
const VIEWS = [["lines", "波形"], ["heatmap", "热力图"], ["stats", "统计表"]];
// 时延路径类型：meta.kind = "latency_path"；波形框与时延框内容互不串选
const isLatency = t => (((DATA[t] || {}).meta || {}).kind === "latency_path");
const latStatOf = t => ((DATA[t] || {}).meta || {}).lat_stat || "";  // RdLat/WrLat
const LAT_MEAN = "__lat_nanmean__";   // 时延框的 nanmean 虚拟类型前缀
const LAT_MEAN_FOR = {};              // 模块组::统计 -> 该组 nanmean 虚拟类型
// 视图也隔离：时延框只有绝对时延堆叠，波形框只有波形/热力图/统计表
const viewsFor = type =>
  isLatency(type) ? [["stack", "时延堆叠"]] : VIEWS.slice();
const TIME_VIEWS = new Set(["lines", "heatmap", "stack"]);
const panels = [];
const topoPanels = [];        // 拓扑框（NoC ring）独立于波形面板
let panelSeq = 0;

// 联动前先做数值比较：plotly_relayout 事件是异步派发的，若靠同步标志位防
// 递归，事件触发时标志已复位，A->B->A 会形成乒乓循环导致页面挂死。
function sameRange(a, b) {
  if (a === null || b === null) return a === b;
  return Math.abs(a[0] - b[0]) <= 1e-9 && Math.abs(a[1] - b[1]) <= 1e-9;
}

function p(v) { return isFinite(v) ? Number(v.toPrecision(6)) : "NaN"; }

// 当前统计口径下的可见模块集合（排除 Σ 曲线自身；热力图为全部模块）
function visibleModules(pn, d) {
  const gd = document.getElementById(pn.id);
  const visible = new Set();
  if (pn.view === "lines" && gd.data)
    gd.data.forEach(t => {
      if (String(t.name).startsWith("Σ")) return;
      if (t.visible === true || t.visible === undefined) visible.add(t.name);
    });
  else d.modules.forEach(m => visible.add(m));
  return visible;
}

// 逐时间点对给定模块求和（跳过 NaN；全 NaN 的时间点输出 null）
function sumCurve(d, visible) {
  const out = new Array(d.times.length).fill(null);
  for (let ti = 0; ti < d.times.length; ti++) {
    let s = 0, any = false;
    d.modules.forEach((m, mi) => {
      if (!visible.has(m)) return;
      const v = d.values[mi][ti];
      if (v != null && isFinite(v)) { s += v; any = true; }
    });
    out[ti] = any ? s : null;
  }
  return out;
}

// 选框内 Σ 曲线的 均值/最大/最小/样本数（与 sumCurve 同口径）
function sumStatsIn(d, visible, x0, x1) {
  const ys = sumCurve(d, visible);
  let sum = 0, n = 0, mx = -Infinity, mn = Infinity;
  d.times.forEach((t, ti) => {
    if (t < x0 || t > x1) return;
    const v = ys[ti];
    if (v == null || !isFinite(v)) return;
    sum += v; n++;
    if (v > mx) mx = v;
    if (v < mn) mn = v;
  });
  return { n, mean: n ? sum / n : NaN, mx, mn };
}

function sameY(a, b) {
  if (!a || a.length !== b.length) return false;
  for (let i = 0; i < b.length; i++) {
    if ((a[i] == null) !== (b[i] == null)) return false;
    if (a[i] != null && a[i] !== b[i]) return false;
  }
  return true;
}

function means(d) {
  return d.modules.map((m, i) => {
    let s = 0, n = 0;
    d.values[i].forEach(v => { if (v != null && isFinite(v)) { s += v; n++; } });
    return n ? s / n : NaN;
  });
}

function tableHTML(d) {
  const rows = d.modules.map((m, i) => {
    const vs = d.values[i].filter(v => v != null && isFinite(v));
    const n = vs.length;
    const mean = n ? vs.reduce((a, b) => a + b, 0) / n : NaN;
    const mx = n ? Math.max(...vs) : NaN;
    const mn = n ? Math.min(...vs) : NaN;
    const varr = n > 1 ? vs.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1) : NaN;
    return { m, n, mean, mx, mn, varr, nanpct: 1 - n / d.times.length };
  }).sort((a, b) => (b.mean || 0) - (a.mean || 0));
  const head = ["模块", "窗口数", "均值", "最大", "最小", "方差", "NaN占比"]
    .map(c => `<th>${c}</th>`).join("");
  const body = rows.map(r =>
    `<tr><td>${r.m}</td><td>${r.n}</td><td>${p(r.mean)}</td><td>${p(r.mx)}</td>` +
    `<td>${p(r.mn)}</td><td>${p(r.varr)}</td><td>${(r.nanpct * 100).toFixed(1)}%</td></tr>`
  ).join("");
  return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
}

function addPanel(type, view, sumOn = false) {
  const id = "p" + panelSeq++;
  const wrap = document.createElement("div");
  wrap.className = "panel";
  wrap.innerHTML =
    `<div class="panel-head">` +
    `<select class="p-mod" title="模块"></select>` +
    `<select class="p-latstat" title="时延统计" style="display:none"></select>` +
    `<select class="p-stat" title="统计项"></select>` +
    `<select class="p-view"></select>` +
    `<label class="p-sum" title="显示当前可见模块的逐时间点求和曲线"><input type="checkbox"> Σ 求和曲线</label>` +
    `<button class="p-close" title="移除面板">×</button>` +
    `<span class="p-info"></span></div>` +
    `<div class="p-chart" id="${id}"></div>` +
    `<div class="p-table"></div>` +
    `<div class="p-stats" id="${id}-stats"></div>`;
  document.getElementById("panels").appendChild(wrap);

  const panel = { id, type, view, xrange: null, selrange: null, showSum: sumOn,
                  wired: false, el: wrap, latency: isLatency(type) };
  // 波形框/时延框互不串类型；NoC 类型只由拓扑框展示，不进波形框
  const kindOk = t => panel.latency
    ? isLatency(t)
    : (!isLatency(t) && !ringShaped(DATA[t]));
  const modSel = wrap.querySelector(".p-mod");
  [...new Set(ALL_TYPES.filter(kindOk).map(moduleOf))].sort().forEach(m => {
    const o = document.createElement("option");
    o.value = o.textContent = m;
    modSel.appendChild(o);
  });
  const statSel = wrap.querySelector(".p-stat");
  const latStatSel = wrap.querySelector(".p-latstat");
  if (panel.latency) latStatSel.style.display = "";
  const fillLatStats = mod => {       // 时延框第二级：该模块组的时延统计（RdLat/WrLat）
    if (!panel.latency) return;
    latStatSel.innerHTML = "";
    [...new Set(ALL_TYPES.filter(t => kindOk(t) && moduleOf(t) === mod)
                           .map(latStatOf))].forEach(s => {
      const o = document.createElement("option");
      o.value = s;
      o.textContent = s || "默认";
      latStatSel.appendChild(o);
    });
  };
  const fillStats = (mod, stat) => {
    statSel.innerHTML = "";
    const meanKey = LAT_MEAN_FOR[mod + "|" + stat];
    if (panel.latency && meanKey) {   // 该统计下的 nanmean（忽略 NaN）
      const o = document.createElement("option");
      o.value = meanKey;
      o.textContent = "nanmean";
      statSel.appendChild(o);
    }
    ALL_TYPES.filter(t => kindOk(t) && moduleOf(t) === mod &&
                          (!panel.latency || latStatOf(t) === stat)).forEach(t => {
      const o = document.createElement("option");
      o.value = t;
      o.textContent = (DATA[t].meta && DATA[t].meta.label) || t;
      statSel.appendChild(o);
    });
  };
  const initMod = moduleOf(type);
  modSel.value = initMod;
  fillLatStats(initMod);
  if (panel.latency) latStatSel.value = latStatOf(type);
  fillStats(initMod, latStatOf(type));
  statSel.value = type;
  const vsel = wrap.querySelector(".p-view");
  const fillViews = () => {          // 时延路径类型多一个「时延堆叠」视图
    vsel.innerHTML = "";
    const vs = viewsFor(panel.type);
    vs.forEach(([v, name]) => {
      const o = document.createElement("option");
      o.value = v; o.textContent = name;
      vsel.appendChild(o);
    });
    if (!vs.some(([v]) => v === panel.view)) panel.view = "lines";
    vsel.value = panel.view;
  };
  fillViews();
  panels.push(panel);
  modSel.onchange = () => {           // 切模块：统计/核列表跟随刷新
    fillLatStats(modSel.value);
    const stat = panel.latency ? (latStatSel.options[0] || {}).value || "" : "";
    fillStats(modSel.value, stat);
    statSel.value = statSel.options[0].value;
    panel.type = statSel.value; panel.xrange = null; panel.selrange = null;
    fillViews();
    renderPanel(panel);
  };
  latStatSel.onchange = () => {
    fillStats(modSel.value, latStatSel.value);
    statSel.value = statSel.options[0].value;
    panel.type = statSel.value; panel.xrange = null; panel.selrange = null;
    renderPanel(panel);
  };
  statSel.onchange = () => {
    panel.type = statSel.value; panel.xrange = null; panel.selrange = null;
    fillViews();
    renderPanel(panel);
  };
  vsel.onchange = () => {
    panel.view = vsel.value; panel.xrange = null; panel.selrange = null;
    renderPanel(panel);
  };
  const sumCb = wrap.querySelector(".p-sum input");
  sumCb.checked = sumOn;              // 首屏 SMMU.TotalBw 默认打开求和曲线
  sumCb.onchange = e => {
    panel.showSum = e.target.checked;
    renderPanel(panel);
  };
  wrap.querySelector(".p-close").onclick = () => removePanel(panel);
  updateCloseButtons();
  renderPanel(panel);
}

function removePanel(panel) {
  Plotly.purge(panel.id);
  panel.el.remove();
  panels.splice(panels.indexOf(panel), 1);
  updateCloseButtons();
}

function updateCloseButtons() {
  // 任意面板都可删除（含第一个），只在仅剩一个面板时隐藏 ×
  const show = panels.length + topoPanels.length > 1;
  panels.forEach(pn =>
    pn.el.querySelector(".p-close").style.display = show ? "" : "none");
  topoPanels.forEach(tp =>
    tp.el.querySelector(".p-close").style.display = show ? "" : "none");
}

function renderPanel(pn) {
  const d = DATA[pn.type];
  const chartEl = document.getElementById(pn.id);
  const tblEl = pn.el.querySelector(".p-table");
  pn.el.querySelector(".p-sum").style.display = pn.view === "lines" ? "" : "none";
  pn.el.querySelector(".p-info").textContent =
    `${d.modules.length} 模块 × ${d.times.length} 窗口` +
    ` ｜ ${(d.meta && d.meta.unit) || ""}`;
  if (pn.view === "stats") {
    Plotly.purge(pn.id);
    pn.wired = false;
    chartEl.style.display = "none";
    tblEl.style.display = "block";
    tblEl.innerHTML = tableHTML(d);
    setStats(pn, "统计表为全量口径；区间统计请切换到波形/热力图视图");
    return;
  }
  tblEl.style.display = "none";
  tblEl.innerHTML = "";
  chartEl.style.display = "block";
  const layout = { margin: { t: 30, r: 20, b: 42, l: 60 },
                   legend: { itemsizing: "trace" } };
  let traces;
  if (pn.view === "lines") {
    const mu = means(d);
    const order = d.modules.map((m, i) => [m, mu[i]]).sort((a, b) => b[1] - a[1]);
    const top = new Set(order.slice(0, TOPN).map(x => x[0]));
    traces = d.modules.map((m, i) => ({
      x: d.times, y: d.values[i], name: m, type: "scatter", mode: "lines",
      line: { width: 1 }, visible: top.has(m) ? true : "legendonly",
    }));
    layout.hovermode = "x unified";
    layout.xaxis = { title: { text: "时间" } };
    layout.yaxis = { title: { text: (d.meta && d.meta.unit) || "值" } };
    layout.dragmode = "select";       // 框选 = 测量，不缩放
    layout.selectdirection = "h";     // 只取水平（时间）区间
    layout.height = 460;              // 显式固定高度：避免视图切换后高度残留
    if (pn.showSum) {                  // Σ 曲线：初始按默认可见模块集求和
      traces.push({ x: d.times, y: sumCurve(d, top), name: "Σ 可见模块",
                    type: "scatter", mode: "lines",
                    line: { width: 2.5, color: "#d62728" } });
    }
  } else if (pn.view === "stack") {   // 时延路径：绝对时延堆叠，顶边=端到端平均时延
    const unit = (d.meta && d.meta.unit) || "";
    traces = d.modules.map((m, i) => ({
      x: d.times, y: d.values[i], name: m, type: "scatter", mode: "lines",
      stackgroup: "one", line: { width: 0.5 },
      hovertemplate: `%{fullData.name}: %{y:.1f} ${unit}<extra></extra>`,
    }));
    layout.hovermode = "x unified";
    layout.xaxis = { title: { text: "时间" } };
    layout.yaxis = { title: { text: `平均时延 (${unit})` } };
    layout.dragmode = "select";
    layout.selectdirection = "h";
    layout.height = 460;
  } else {  // heatmap
    traces = [{ type: "heatmap", x: d.times, y: d.modules, z: d.values,
                colorbar: { title: "值" },
                hovertemplate: "%{y}<br>t=%{x}<br>值=%{z}<extra></extra>" }];
    layout.xaxis = { title: { text: "时间" } };
    layout.yaxis = { title: { text: "模块" } };
    layout.dragmode = "select";
    layout.selectdirection = "h";
    layout.height = 460;
  }
  if (TIME_VIEWS.has(pn.view) && pn.xrange)
    layout.xaxis.range = pn.xrange;   // 重渲染（切Σ/切类型/切视图）时保留缩放
  chartEl.style.height = "";          // 清除 plotly 上次渲染写下的内联高度残留
  Plotly.react(pn.id, traces, layout,
               { responsive: true, displaylogo: false, scrollZoom: false });
  if (!pn.wired) {
    pn.wired = true;
    const gd = document.getElementById(pn.id);
    gd.on("plotly_selected", ev => onSelect(pn, ev));   // 框选 -> 选区统计
    gd.on("plotly_deselect", () => onDeselect(pn));     // 双击 -> 全区间基线
    gd.on("plotly_relayout", ev => onRelayout(pn, ev));  // 缩放 -> 仅时间轴联动
    gd.on("plotly_restyle", () => {                     // 图例开关 -> 刷新口径与 Σ
      updateSum(pn);
      updateStats(pn);
    });
  }
  updateStats(pn);
}

function updateSum(pn) {
  if (!pn.showSum || pn.view !== "lines") return;
  const gd = document.getElementById(pn.id);
  const idx = gd.data.findIndex(t => String(t.name).startsWith("Σ"));
  if (idx < 0) return;
  const next = sumCurve(DATA[pn.type], visibleModules(pn, DATA[pn.type]));
  if (sameY(gd.data[idx].y, next)) return;   // restyle 内再 restyle 会循环，值不变则跳过
  Plotly.restyle(gd, { y: [next] }, [idx]);
}

function onSelect(pn, ev) {
  if (ev && ev.range && ev.range.x) pn.selrange = [ev.range.x[0], ev.range.x[1]];
  updateStats(pn);
}

function onDeselect(pn) {
  pn.selrange = null;
  updateStats(pn);
}

function onRelayout(pn, ev) {
  const r = ev["xaxis.range"];
  if (r) pn.xrange = [r[0], r[1]];
  else if ("xaxis.autorange" in ev) pn.xrange = null;
  else return;
  panels.forEach(o => {           // 统计条只跟随选区，缩放仅用于联动时间轴
    if (o === pn || !TIME_VIEWS.has(o.view)) return;
    if (sameRange(o.xrange, pn.xrange)) return;   // 已一致则跳过，断开乒乓
    if (pn.xrange) Plotly.relayout(o.id, { "xaxis.range": pn.xrange });
    else Plotly.relayout(o.id, { "xaxis.autorange": true });
  });
}

function updateStats(pn) {
  if (!TIME_VIEWS.has(pn.view)) {
    setStats(pn, "统计表为全量口径；区间统计请切换到波形/热力图视图");
    return;
  }
  if (pn.view === "stack") {          // 堆叠口径：各路径平均时延之和 = 端到端平均时延
    const d0 = DATA[pn.type];
    const [sx0, sx1] = pn.selrange || [d0.times[0], d0.times[d0.times.length - 1]];
    const slab = pn.selrange ? "选区" : "全区间";
    const vis = visibleModules(pn, d0);
    let s = 0, cnt = 0, smx = -Infinity, smn = Infinity;
    d0.times.forEach((t, ti) => {
      if (t < sx0 || t > sx1) return;
      let total = 0, any = false;
      d0.modules.forEach((m, mi) => {
        if (!vis.has(m)) return;
        const v = d0.values[mi][ti];
        if (v != null && isFinite(v)) { total += v; any = true; }
      });
      if (!any) return;
      s += total; cnt++;
      if (total > smx) smx = total;
      if (total < smn) smn = total;
    });
    if (!cnt) { setStats(pn, `${slab} [${p(sx0)}, ${p(sx1)}] 内无有效数据`); return; }
    setStats(pn,
      `${slab} [${p(sx0)}, ${p(sx1)}]: 总时延 均值 <b>${p(s / cnt)}</b>` +
      ` · 最大 <b>${p(smx)}</b> · 最小 <b>${p(smn)}</b> · 时间点 <b>${cnt}</b>` +
      `（可见路径 ${vis.size} 条） ｜ 堆叠高度=各路径平均时延之和`);
    return;
  }
  const d = DATA[pn.type];
  const [x0, x1] = pn.selrange ||
    [d.times[0], d.times[d.times.length - 1]];
  const label = pn.selrange ? "选区" : "全区间";
  const visible = visibleModules(pn, d);
  let sum = 0, n = 0, mx = -Infinity, mn = Infinity;
  d.modules.forEach((m, mi) => {
    if (!visible.has(m)) return;
    for (let ti = 0; ti < d.times.length; ti++) {
      const t = d.times[ti];
      if (t < x0 || t > x1) continue;
      const v = d.values[mi][ti];
      if (v == null || !isFinite(v)) continue;
      sum += v; n++;
      if (v > mx) mx = v;
      if (v < mn) mn = v;
    }
  });
  if (!n) { setStats(pn, `${label} [${p(x0)}, ${p(x1)}] 内无有效数据`); return; }
  let html =
    `${label} [${p(x0)}, ${p(x1)}]: 均值 <b>${p(sum / n)}</b>` +
    ` · 最大 <b>${p(mx)}</b> · 最小 <b>${p(mn)}</b> · 样本数 <b>${n}</b>` +
    (pn.view === "lines" ? `（仅可见曲线 ${visible.size} 条）` : "（全部模块）");
  if (pn.view === "lines" && pn.showSum) {   // 显示 Σ 曲线时单独追加一段
    const s = sumStatsIn(d, visible, x0, x1);
    html += s.n
      ? ` ｜ Σ: 均值 <b>${p(s.mean)}</b> · 最大 <b>${p(s.mx)}</b>` +
        ` · 最小 <b>${p(s.mn)}</b> · 样本数 <b>${s.n}</b>`
      : " ｜ Σ: 选区内无有效数据";
  }
  html += " ｜ 拖拽框选测量区间，双击清除（缩放用工具栏按钮）";
  setStats(pn, html);
}

function setStats(pn, html) {
  document.getElementById(pn.id + "-stats").innerHTML = html;
}

// ------------------------------------------------------------ 拓扑框（NoC ring）
const RING_METRICS = {
  upIn: "noc_UpInFail", upOut: "noc_UpOutFail",
  downIn: "noc_DownInFail", downOut: "noc_DownOutFail",
  left: "noc_left_util", right: "noc_right_util",
};
const ringShaped = d => d.modules.length > 0 && d.modules.every(m => {
  const x = String(m).split("_");
  return x.length === 3 && x.every(Boolean);
});
const RING_TYPES = ALL_TYPES.filter(t => ringShaped(DATA[t]));

function parseNoc(d) {
  const rings = new Set(), channels = new Set(), map = {};
  for (let i = 0; i < d.modules.length; i++) {
    const t = String(d.modules[i]).split("_");
    if (t.length !== 3 || t.some(x => !x)) return null;
    rings.add(t[0]); channels.add(t[1]);
    map[t[0] + "_" + t[1] + "_" + t[2]] = i;
  }
  const csByCh = {};
  [...channels].sort().forEach(c => {
    const s = new Set();
    d.modules.forEach(m => { const t = String(m).split("_"); if (t[1] === c) s.add(t[2]); });
    csByCh[c] = [...s].sort();
  });
  return { rings: [...rings].sort(), channels: [...channels], csByCh, map };
}

function nocRange(d) {
  let mn = Infinity, mxv = -Infinity;
  d.modules.forEach((m, mi) => d.values[mi].forEach(v => {
    if (v == null || !isFinite(v)) return;
    if (v < mn) mn = v;
    if (v > mxv) mxv = v;
  }));
  return { min: isFinite(mn) ? mn : 0,
           maxVal: isFinite(mxv) ? mxv : (isFinite(mn) ? mn : 1) };
}

function cellColor(v, sc) {
  if (v == null || !isFinite(v)) return "";
  const k = sc.maxVal === sc.min ? 0
    : Math.max(0, Math.min(1, (v - sc.min) / (sc.maxVal - sc.min)));
  const r = Math.round(255 + (214 - 255) * k);
  const g = Math.round(255 + (39 - 255) * k);
  const b = Math.round(255 + (40 - 255) * k);
  return `rgb(${r},${g},${b})`;
}

// 方框/箭头的紧凑数值：<10 保留 2 位小数，<100 保留 1 位，更大取整
function fmtShort(v) {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 100) return String(Math.round(v));
  if (a >= 10) return String(Math.round(v * 10) / 10);
  return String(Math.round(v * 100) / 100);
}

function nocCellAt(d, rowIdx, view) {   // view: {i1, i2} 选点或窗口均值
  if (rowIdx == null) return null;
  if (view.i2 > view.i1) {
    const x0 = d.times[view.i1], x1 = d.times[view.i2];
    let s = 0, n = 0;
    for (let ti = 0; ti < d.times.length; ti++) {
      const t = d.times[ti];
      if (t < x0 || t > x1) continue;
      const v = d.values[rowIdx][ti];
      if (v != null && isFinite(v)) { s += v; n++; }
    }
    return n ? s / n : null;
  }
  return d.values[rowIdx][view.i1];
}

const _ringIdx = {}, _ringSc = {};
function ringIdx(type, name) {
  if (!DATA[type]) return -1;
  if (!_ringIdx[type]) {
    const m = {};
    DATA[type].modules.forEach((x, i) => m[x] = i);
    _ringIdx[type] = m;
  }
  return _ringIdx[type][name] !== undefined ? _ringIdx[type][name] : -1;
}
function ringScale(type) {
  if (!type || !DATA[type]) return null;
  if (!_ringSc[type]) _ringSc[type] = nocRange(DATA[type]);
  return _ringSc[type];
}
function metricVal(type, name, view) {
  const mi = ringIdx(type, name);
  return mi < 0 ? null : nocCellAt(DATA[type], mi, view);
}

function addTopoPanel() {
  const timeType = RING_TYPES[0];
  if (!timeType) return;
  const parsed = parseNoc(DATA[timeType]);
  if (!parsed) return;
  const state = { channel: parsed.channels[0], i1: 0, i2: 0,
                  timer: null, die: "die0" };
  const dieOf = r => RING_DIE[r] || "die0";   // 未声明 die 的 ring 归 die0
  const wrap = document.createElement("div");
  wrap.className = "panel";
  wrap.innerHTML =
    `<div class="panel-head">` +
    `<span class="p-info" style="color:#4a90d9;font-weight:600">ring 拓扑</span>` +
    `<select class="p-ch" title="通道"></select>` +
    `<span class="p-info p-meta"></span>` +
    `<span class="p-zoom" title="缩放拓扑图">` +
    `<button class="p-zout">−</button><span class="p-zval">100%</span>` +
    `<button class="p-zin">＋</button><button class="p-zreset">还原</button></span>` +
    `<button class="p-close" title="移除面板">×</button></div>` +
    `<div class="ring-filter"><span>Die：</span>` +
    `<select class="p-die" title="die"></select></div>` +
    `<div class="time-inputs">时间窗：` +
    `<input class="p-t0" type="number" step="any" title="开始时间">` +
    ` ～ <input class="p-t1" type="number" step="any" title="结束时间"></div>` +
    `<div class="p-noc p-topo-bar" style="display:flex">` +
    `<div class="n-track p-track" title="单击=选点；按住拖动或拖动手柄=选时间窗">` +
    `<div class="n-fill"></div><div class="n-h n-h1"></div><div class="n-h n-h2"></div></div>` +
    `<span class="n-time p-time"></span>` +
    `<button class="n-play p-play">▶ 回放</button></div>` +
    `<div class="p-topo"><div class="p-topo-inner"></div></div>` +
    `<div class="p-stats p-topo-stats"></div>`;
  document.getElementById("panels").appendChild(wrap);
  const $ = sel => wrap.querySelector(sel);
  const topoHost = $(".p-topo");
  const topoInner = $(".p-topo-inner");
  const dieSel = $(".p-die");
  const chSel = $(".p-ch");
  const track = $(".p-track");
  let refs = null, builtCh = null;

  const entry = { el: wrap };
  topoPanels.push(entry);
  updateCloseButtons();

  parsed.channels.forEach(c => {           // 通道下拉
    const o = document.createElement("option");
    o.value = o.textContent = c;
    chSel.appendChild(o);
  });
  chSel.value = state.channel;
  chSel.onchange = () => { state.channel = chSel.value; refs = null; update(); };

  function fillDies() {                    // die 下拉：默认 die0
    const dies = [...new Set(parsed.rings.map(dieOf))];
    if (!dies.length) return;
    state.die = dies.includes("die0") ? "die0" : dies[0];
    dieSel.innerHTML = "";
    dies.forEach(d => {
      const o = document.createElement("option");
      o.value = o.textContent = d;
      dieSel.appendChild(o);
    });
    dieSel.value = state.die;
    dieSel.onchange = () => { state.die = dieSel.value; refs = null; update(); };
  }

  function buildTopo() {
    topoInner.innerHTML = "";
    refs = null; builtCh = state.channel;
    const csList = parsed.csByCh[state.channel] || [];
    const n = csList.length;
    const legend = document.createElement("div");
    legend.className = "topo-legend";
    legend.textContent =
      "角标：左上 UpInFail ｜ 右上 UpOutFail ｜ 左下 DownInFail ｜ 右下 DownOutFail" +
      " ｜ 箭头 → right_util、← left_util（颜色=强度，悬停看数值）";
    topoInner.appendChild(legend);
    const linkHTML = tag =>
      `<div class="lnk ${tag}">` +
      `<div class="arr ${tag}-f"><span class="ag">▶</span><span class="av"></span></div>` +
      `<div class="arr ${tag}-b"><span class="ag">◀</span><span class="av"></span></div></div>`;
    refs = parsed.rings.filter(r => dieOf(r) === state.die).map(r => {
      let inner = `<div class="topo-name">${r}</div><div class="topo-strip">` +
        linkHTML("endl");
      csList.forEach((cs, i) => {
        inner += `<div class="tbox"><div class="tcs">${cs}</div>` +
          `<div class="tip">${NODE_NAMES[r + "_" + cs] || ""}</div>` +
          `<div class="tgrid">` +
          `<span class="tcell c-tl"></span><span class="tcell c-tr"></span>` +
          `<span class="tcell c-bl"></span><span class="tcell c-br"></span>` +
          `</div></div>`;
        inner += i < n - 1 ? linkHTML("mid") : linkHTML("endr");
      });
      const row = document.createElement("div");
      row.className = "topo-ring";
      row.innerHTML = inner + "</div>";
      topoInner.appendChild(row);
      return {
        ring: r,
        corners: [...row.querySelectorAll(".tbox")].map(b => ({
          tl: b.querySelector(".c-tl"), tr: b.querySelector(".c-tr"),
          bl: b.querySelector(".c-bl"), br: b.querySelector(".c-br"),
        })),
        mids: [...row.querySelectorAll(".lnk.mid")].map(l => ({
          fwd: l.querySelector(".mid-f"), bwd: l.querySelector(".mid-b"),
        })),
        endL: { fwd: row.querySelector(".endl-f"), bwd: row.querySelector(".endl-b") },
        endR: { fwd: row.querySelector(".endr-f"), bwd: row.querySelector(".endr-b") },
      };
    });
  }

  function update() {                      // 刷新四角数值、箭头与时间条
    if (builtCh !== state.channel || refs === null) buildTopo();
    const ch = state.channel;
    const setCell = (el, type, name) => {
      const v = metricVal(type, name, state);
      el.textContent = fmtShort(v);
      const sc = ringScale(type);
      el.style.background = sc ? cellColor(v, sc) : "";
      el.title = `${name} ${type}=${v == null || !isFinite(v) ? "-" : p(v)}`;
    };
    const setArrow = (el, type, name, dir) => {
      const v = metricVal(type, name, state);
      el.querySelector(".av").textContent = fmtShort(v);
      const sc = ringScale(type);
      el.style.background = sc ? cellColor(v, sc) : "";
      el.title = `${dir} ${name} ${type}=${v == null || !isFinite(v) ? "-" : p(v)}`;
    };
    refs.forEach(rf => {
      const r = rf.ring;
      const csList = parsed.csByCh[ch] || [];
      const n = csList.length;
      const mod = cs => `${r}_${ch}_${cs}`;
      csList.forEach((cs, i) => {
        const c = rf.corners[i];
        if (!c) return;
        setCell(c.tl, RING_METRICS.upIn, mod(cs));
        setCell(c.tr, RING_METRICS.upOut, mod(cs));
        setCell(c.bl, RING_METRICS.downIn, mod(cs));
        setCell(c.br, RING_METRICS.downOut, mod(cs));
      });
      for (let i = 0; i + 1 < n; i++) {
        setArrow(rf.mids[i].fwd, RING_METRICS.right, mod(csList[i]), "→");
        setArrow(rf.mids[i].bwd, RING_METRICS.left, mod(csList[i + 1]), "←");
      }
      if (n) {                            // 环绕链路在带两端均可见
        setArrow(rf.endL.bwd, RING_METRICS.left, mod(csList[0]), "←");
        setArrow(rf.endL.fwd, RING_METRICS.right, mod(csList[n - 1]), "→");
        setArrow(rf.endR.fwd, RING_METRICS.right, mod(csList[n - 1]), "→");
        setArrow(rf.endR.bwd, RING_METRICS.left, mod(csList[0]), "←");
      }
    });
    const times = DATA[timeType].times;
    const nT = times.length;
    const frac = i => nT > 1 ? i / (nT - 1) : 0;
    $(".n-h1").style.left = `calc(${frac(state.i1) * 100}% - 6px)`;
    $(".n-h2").style.left = `calc(${frac(state.i2) * 100}% - 6px)`;
    const fill = $(".n-fill");
    fill.style.left = `${frac(state.i1) * 100}%`;
    fill.style.width = `${(frac(state.i2) - frac(state.i1)) * 100}%`;
    const isWin = state.i2 > state.i1;
    $(".p-time").textContent = isWin
      ? `窗口 [${p(times[state.i1])}, ${p(times[state.i2])}]`
      : `t=${p(times[state.i1])}`;
    t0El.value = p(times[state.i1]);       // 输入框与拖拽条双向同步
    t1El.value = p(times[state.i2]);
    $(".p-meta").textContent =
      `${parsed.rings.length} ring × ${(parsed.csByCh[ch] || []).length} CS`;
    $(".p-topo-stats").innerHTML =
      `拓扑图 ｜ 通道 ${ch} ｜ ${isWin
        ? `窗口 [${p(times[state.i1])}, ${p(times[state.i2])}] 均值（单击时间条回到选点）`
        : `时刻 t=${p(times[state.i1])}（按住拖动时间条可选窗口）`}`;
  }

  const h1El = $(".n-h1"), h2El = $(".n-h2"), play = $(".p-play");
  const idxAt = ev => {                    // 时间条拖拽（与独立页面同一交互）
    const r = track.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    return Math.round(f * (Math.max(1, DATA[timeType].times.length) - 1));
  };
  let drag = null;
  const move = ev => {
    const i = idxAt(ev);
    if (drag === "h1") state.i1 = Math.min(i, state.i2);
    else if (drag === "h2") state.i2 = Math.max(i, state.i1);
    else if (drag === "track") state.i2 = Math.max(i, state.i1);
    update();
  };
  track.addEventListener("pointerdown", ev => {
    const i = idxAt(ev);
    if (ev.target === h1El) drag = "h1";
    else if (ev.target === h2El) drag = "h2";
    else { drag = "track"; state.i1 = state.i2 = i; }
    track.setPointerCapture(ev.pointerId);
    move(ev);
    ev.preventDefault();
  });
  track.addEventListener("pointermove", ev => { if (drag) move(ev); });
  track.addEventListener("pointerup", () => { drag = null; });
  play.onclick = () => {
    const nT = DATA[timeType].times.length;
    if (state.timer) {
      clearInterval(state.timer); state.timer = null;
      play.textContent = "▶ 回放";
    } else {
      state.timer = setInterval(() => {
        state.i1 = state.i2 = (state.i2 + 1) % Math.max(1, nT);
        update();
      }, 150);
      play.textContent = "⏸ 暂停";
    }
  };
  $(".p-close").onclick = () => {
    if (state.timer) clearInterval(state.timer);
    wrap.remove();
    topoPanels.splice(topoPanels.indexOf(entry), 1);
    updateCloseButtons();
  };

  // 时间输入框：填写后按最近采样点落位（起止顺序自动纠正）
  const t0El = $(".p-t0"), t1El = $(".p-t1");
  const nearestIdx = v => {
    let best = 0, bd = Infinity;
    DATA[timeType].times.forEach((x, i) => {
      const d = Math.abs(x - v);
      if (d < bd) { bd = d; best = i; }
    });
    return best;
  };
  const applyTimeInputs = () => {
    const v0 = parseFloat(t0El.value), v1 = parseFloat(t1El.value);
    if (!isFinite(v0) || !isFinite(v1)) return;
    let a = nearestIdx(v0), b = nearestIdx(v1);
    if (a > b) { const t = a; a = b; b = t; }
    state.i1 = a; state.i2 = b;
    update();
  };
  t0El.onchange = applyTimeInputs;
  t1El.onchange = applyTimeInputs;

  // 缩放：只作用于拓扑图内容，「还原」回到 100%
  let zoom = 1;
  const zVal = $(".p-zval");
  const setZoom = z => {
    zoom = Math.max(0.5, Math.min(3, z));
    zVal.textContent = Math.round(zoom * 100) + "%";
    topoInner.style.zoom = zoom;
  };
  $(".p-zin").onclick = () => setZoom(zoom + 0.1);
  $(".p-zout").onclick = () => setZoom(zoom - 0.1);
  $(".p-zreset").onclick = () => setZoom(1);
  setZoom(1);

  fillDies();
  update();
}

const topoBtn = document.getElementById("add-topo-panel");
if (RING_TYPES.length) topoBtn.onclick = () => addTopoPanel();
else topoBtn.style.display = "none";

// 时延框：类型 meta.kind = "latency_path"（每 AIC 一份路径矩阵）
const LATENCY_TYPES = ALL_TYPES.filter(isLatency);
// 波形框只收普通类型：时延走时延框，NoC（R_C_CS 行标）走拓扑框
const WAVE_TYPES = ALL_TYPES.filter(t =>
  !isLatency(t) && !ringShaped(DATA[t]));
// nanmean：每个（模块组, 时延统计）内逐元素取均值（忽略 NaN），行标沿用组内首个
let LAT_MEAN_DEFAULT = null;
if (LATENCY_TYPES.length) {
  [...new Set(LATENCY_TYPES.map(moduleOf))].forEach(mod => {
    const modTypes = LATENCY_TYPES.filter(t => moduleOf(t) === mod);
    [...new Set(modTypes.map(latStatOf))].forEach(stat => {
      const types = modTypes.filter(t => latStatOf(t) === stat);
      const base = DATA[types[0]];
      const key = LAT_MEAN + "|" + mod + "|" + stat;
      LAT_MEAN_FOR[mod + "|" + stat] = key;
      if (!LAT_MEAN_DEFAULT) LAT_MEAN_DEFAULT = key;
      DATA[key] = {
        meta: base.meta, times: base.times, modules: base.modules,
        values: base.modules.map((m, mi) => base.times.map((t, ti) => {
          let s = 0, n = 0;
          types.forEach(tp => {
            const d = DATA[tp], i = d.modules.indexOf(m);
            if (i < 0 || ti >= d.times.length) return;
            const v = d.values[i][ti];
            if (v != null && isFinite(v)) { s += v; n++; }
          });
          return n ? s / n : NaN;
        })),
      };
    });
  });
}
const latBtn = document.getElementById("add-latency-panel");
if (LATENCY_TYPES.length)
  latBtn.onclick = () => addPanel(LAT_MEAN_DEFAULT, "stack");   // 默认 nanmean
else latBtn.style.display = "none";

const waveBtn = document.getElementById("add-panel");
let waveSeq = 0;
if (WAVE_TYPES.length)
  waveBtn.onclick = () => addPanel(WAVE_TYPES[waveSeq++ % WAVE_TYPES.length], "lines");
else waveBtn.style.display = "none";
// 初始面板：优先 SMMU.TotalBw；无则取第一个普通类型；只有时延数据时退回时延框
const DEFAULT_WAVE = "SMMU.TotalBw";
const initType = WAVE_TYPES.length
  ? (WAVE_TYPES.includes(DEFAULT_WAVE) ? DEFAULT_WAVE : WAVE_TYPES[0])
  : LAT_MEAN_DEFAULT;
if (initType)
  addPanel(initType, WAVE_TYPES.length ? "lines" : "stack",
           initType === DEFAULT_WAVE);   // 首屏 SMMU.TotalBw 默认显示 Σ 曲线
"""

_JS_RING_APP = r"""
const DATA = __DATA__;
const NODE_NAMES = __NODES__;          // {"R00_CS00": "HBMC", ...} 槽位 IP 名
const ALL_TYPES = Object.keys(DATA).sort();
const moduleOf = t => (DATA[t].meta && DATA[t].meta.module) || "默认";
const st = { type: ALL_TYPES[0], channel: null, i1: 0, i2: 0, timer: null,
             parsed: null, mode: "topo", hidden: new Set() };

const shownRings = parsed => parsed.rings.filter(r => !st.hidden.has(r));

// 拓扑图四角/箭头对应的指标类型（缺失时该位置显示 "-"）
const METRICS = {
  upIn: "noc_UpInFail", upOut: "noc_UpOutFail",
  downIn: "noc_DownInFail", downOut: "noc_DownOutFail",
  left: "noc_left_util", right: "noc_right_util",
};

function p(v) { return isFinite(v) ? Number(v.toPrecision(6)) : "NaN"; }

// 拓扑图方框/箭头的紧凑数值：<10 保留 2 位小数，<100 保留 1 位，更大取整
function fmtShort(v) {
  if (v == null || !isFinite(v)) return "-";
  const a = Math.abs(v);
  if (a >= 100) return String(Math.round(v));
  if (a >= 10) return String(Math.round(v * 10) / 10);
  return String(Math.round(v * 100) / 100);
}

function parseNoc(d) {
  const rings = new Set(), channels = new Set(), map = {};
  for (let i = 0; i < d.modules.length; i++) {
    const t = String(d.modules[i]).split("_");
    if (t.length !== 3 || t.some(x => !x)) return null;
    rings.add(t[0]); channels.add(t[1]);
    map[t[0] + "_" + t[1] + "_" + t[2]] = i;
  }
  const csByCh = {};
  [...channels].sort().forEach(c => {
    const s = new Set();
    d.modules.forEach(m => { const t = String(m).split("_"); if (t[1] === c) s.add(t[2]); });
    csByCh[c] = [...s].sort();
  });
  // 通道按模块出现顺序（Req/Snp/Rsp/Dat），不按字典序
  return { rings: [...rings].sort(), channels: [...channels], csByCh, map };
}

function nocDerived(d) {
  let mn = Infinity, mxv = -Infinity;
  d.modules.forEach((m, mi) => d.values[mi].forEach(v => {
    if (v == null || !isFinite(v)) return;
    if (v < mn) mn = v;
    if (v > mxv) mxv = v;
  }));
  return { min: isFinite(mn) ? mn : 0,
           maxVal: isFinite(mxv) ? mxv : (isFinite(mn) ? mn : 1) };
}

function cellColor(v, sc) {
  if (v == null || !isFinite(v)) return "";
  const k = sc.maxVal === sc.min ? 0
    : Math.max(0, Math.min(1, (v - sc.min) / (sc.maxVal - sc.min)));
  const r = Math.round(255 + (214 - 255) * k);
  const g = Math.round(255 + (39 - 255) * k);
  const b = Math.round(255 + (40 - 255) * k);
  return `rgb(${r},${g},${b})`;
}

function nocCell(d, rowIdx) {
  if (rowIdx == null) return null;
  if (st.i2 > st.i1) {                 // 窗口模式：[t(i1), t(i2)] 窗口内均值
    const x0 = d.times[st.i1], x1 = d.times[st.i2];
    let s = 0, n = 0;
    for (let ti = 0; ti < d.times.length; ti++) {
      const t = d.times[ti];
      if (t < x0 || t > x1) continue;
      const v = d.values[rowIdx][ti];
      if (v != null && isFinite(v)) { s += v; n++; }
    }
    return n ? s / n : null;
  }
  return d.values[rowIdx][st.i1];     // 选点模式
}

// ------------------------------------------------------------ 拓扑图（NoC 流量观察）

const _idxCache = {};
function modIdx(type, name) {         // 模块名 -> 该类型数据中的行号
  if (!DATA[type]) return -1;
  if (!_idxCache[type]) {
    const m = {};
    DATA[type].modules.forEach((x, i) => m[x] = i);
    _idxCache[type] = m;
  }
  return _idxCache[type][name] !== undefined ? _idxCache[type][name] : -1;
}
const _scCache = {};
const scOf = type => {                 // 各指标全局色标（min/max），跨时刻可比
  if (!type || !DATA[type]) return null;
  if (!_scCache[type]) _scCache[type] = nocDerived(DATA[type]);
  return _scCache[type];
};
function metricVal(type, name) {      // 按当前选点/窗口取该模块该指标值
  const mi = modIdx(type, name);
  return mi < 0 ? null : nocCell(DATA[type], mi);
}

let topoRefs = null;
let topoBuiltCh = null;

function buildTopo() {                 // 每个 ring 一条横向 CS 方框带，首尾环绕
  const parsed = st.parsed;
  const host = document.getElementById("topo");
  host.innerHTML = "";
  topoRefs = null;
  topoBuiltCh = st.channel;
  if (!parsed) return;
  const legend = document.createElement("div");
  legend.className = "topo-legend";
  legend.textContent =
    "角标：左上 UpInFail ｜ 右上 UpOutFail ｜ 左下 DownInFail ｜ 右下 DownOutFail" +
    " ｜ 箭头 → right_util、← left_util（颜色=强度，悬停看数值）";
  host.appendChild(legend);
  topoRefs = shownRings(parsed).map(r => {
    const csList = parsed.csByCh[st.channel] || [];
    const n = csList.length;
    const linkHTML = tag =>
      `<div class="lnk ${tag}">` +
      `<div class="arr ${tag}-f"><span class="ag">▶</span><span class="av"></span></div>` +
      `<div class="arr ${tag}-b"><span class="ag">◀</span><span class="av"></span></div></div>`;
    let inner = `<div class="topo-name">${r}</div><div class="topo-strip">` +
      linkHTML("endl");                       // 首框左侧：◀ 本CS left_util / ▶ 尾CS right_util
    csList.forEach((cs, i) => {
      inner += `<div class="tbox"><div class="tcs">${cs}</div>` +
        `<div class="tip">${NODE_NAMES[r + "_" + cs] || ""}</div>` +
        `<div class="tgrid">` +
        `<span class="tcell c-tl"></span><span class="tcell c-tr"></span>` +
        `<span class="tcell c-bl"></span><span class="tcell c-br"></span>` +
        `</div></div>`;
      inner += i < n - 1 ? linkHTML("mid")    // 相邻链路：▶ 左CS right_util / ◀ 右CS left_util
                         : linkHTML("endr");  // 尾框右侧：▶ 尾CS right_util / ◀ 首CS left_util
    });
    const row = document.createElement("div");
    row.className = "topo-ring";
    row.innerHTML = inner + "</div>";
    host.appendChild(row);
    return {
      ring: r,
      corners: [...row.querySelectorAll(".tbox")].map(b => ({
        tl: b.querySelector(".c-tl"), tr: b.querySelector(".c-tr"),
        bl: b.querySelector(".c-bl"), br: b.querySelector(".c-br"),
      })),
      mids: [...row.querySelectorAll(".lnk.mid")].map(l => ({
        fwd: l.querySelector(".mid-f"), bwd: l.querySelector(".mid-b"),
      })),
      endL: { fwd: row.querySelector(".endl-f"), bwd: row.querySelector(".endl-b") },
      endR: { fwd: row.querySelector(".endr-f"), bwd: row.querySelector(".endr-b") },
    };
  });
}

function updateTopo() {                // 按当前选点/窗口刷新四角数值与箭头颜色/数值
  if (!topoRefs) return;
  const parsed = st.parsed, ch = st.channel;
  const setCell = (el, type, name) => {
    const v = metricVal(type, name);
    el.textContent = fmtShort(v);
    const sc = scOf(type);
    el.style.background = sc ? cellColor(v, sc) : "";
    el.title = `${name} ${type}=${v == null || !isFinite(v) ? "-" : p(v)}`;
  };
  const setArrow = (el, type, name, dir) => {
    const v = metricVal(type, name);
    el.querySelector(".av").textContent = fmtShort(v);   // 箭头标注利用率数值
    const sc = scOf(type);
    el.style.background = sc ? cellColor(v, sc) : "";
    el.title = `${dir} ${name} ${type}=${v == null || !isFinite(v) ? "-" : p(v)}`;
  };
  topoRefs.forEach(refs => {
    const r = refs.ring;
    const csList = parsed.csByCh[ch] || [];
    const n = csList.length;
    const mod = cs => `${r}_${ch}_${cs}`;
    csList.forEach((cs, i) => {
      const c = refs.corners[i];
      if (!c) return;
      setCell(c.tl, METRICS.upIn, mod(cs));
      setCell(c.tr, METRICS.upOut, mod(cs));
      setCell(c.bl, METRICS.downIn, mod(cs));
      setCell(c.br, METRICS.downOut, mod(cs));
    });
    for (let i = 0; i + 1 < n; i++) {
      setArrow(refs.mids[i].fwd, METRICS.right, mod(csList[i]), "→");
      setArrow(refs.mids[i].bwd, METRICS.left, mod(csList[i + 1]), "←");
    }
    if (n) {                            // 环绕链路在带状图两端均可见
      setArrow(refs.endL.bwd, METRICS.left, mod(csList[0]), "←");
      setArrow(refs.endL.fwd, METRICS.right, mod(csList[n - 1]), "→");
      setArrow(refs.endR.fwd, METRICS.right, mod(csList[n - 1]), "→");
      setArrow(refs.endR.bwd, METRICS.left, mod(csList[0]), "←");
    }
  });
}

function initTimeBar() {
  const track = document.getElementById("track");
  const h1El = document.querySelector(".n-h1");
  const h2El = document.querySelector(".n-h2");
  const playBtn = document.getElementById("play");
  const idxAt = ev => {
    const r = track.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
    return Math.round(frac * (Math.max(1, DATA[st.type].times.length) - 1));
  };
  let drag = null;
  const move = ev => {
    const i = idxAt(ev);
    if (drag === "h1") st.i1 = Math.min(i, st.i2);
    else if (drag === "h2") st.i2 = Math.max(i, st.i1);
    else if (drag === "track") st.i2 = Math.max(i, st.i1);
    update();
  };
  track.addEventListener("pointerdown", ev => {
    const i = idxAt(ev);
    if (ev.target === h1El) drag = "h1";
    else if (ev.target === h2El) drag = "h2";
    else { drag = "track"; st.i1 = st.i2 = i; }
    track.setPointerCapture(ev.pointerId);
    move(ev);
    ev.preventDefault();
  });
  track.addEventListener("pointermove", ev => { if (drag) move(ev); });
  track.addEventListener("pointerup", () => { drag = null; });
  playBtn.onclick = () => {
    const nT = DATA[st.type].times.length;
    if (st.timer) {
      clearInterval(st.timer); st.timer = null;
      playBtn.textContent = "▶ 回放";
    } else {
      st.timer = setInterval(() => {
        st.i1 = st.i2 = (st.i2 + 1) % Math.max(1, nT);
        update();
      }, 150);
      playBtn.textContent = "⏸ 暂停";
    }
  };
}

function update() {
  if (!st.parsed) return;
  const d = DATA[st.type], parsed = st.parsed;
  if (st.mode === "topo") {
    if (topoBuiltCh !== st.channel || topoRefs === null) buildTopo();
    updateTopo();
    document.getElementById("rtable").innerHTML = "";
  } else {
    const sc = nocDerived(d);
    const csList = parsed.csByCh[st.channel] || [];
    const head = "<tr><th></th>" + csList.map(c => `<th>${c}</th>`).join("") + "</tr>";
    const rows = shownRings(parsed).map(r => {
      const cells = csList.map(cs => {
        const mi = parsed.map[r + "_" + st.channel + "_" + cs];
        const v = nocCell(d, mi);
        const txt = v == null || !isFinite(v) ? "-" : p(v);
        return `<td style="background:${cellColor(v, sc)}">${txt}</td>`;
      }).join("");
      return `<tr><td>${r}</td>${cells}</tr>`;
    }).join("");
    document.getElementById("rtable").innerHTML =
      `<table class="noc-table"><thead>${head}</thead><tbody>${rows}</tbody></table>`;
  }

  const nT = d.times.length;
  const frac = i => nT > 1 ? i / (nT - 1) : 0;
  document.querySelector(".n-h1").style.left = `calc(${frac(st.i1) * 100}% - 6px)`;
  document.querySelector(".n-h2").style.left = `calc(${frac(st.i2) * 100}% - 6px)`;
  const fill = document.querySelector(".n-fill");
  fill.style.left = `${frac(st.i1) * 100}%`;
  fill.style.width = `${(frac(st.i2) - frac(st.i1)) * 100}%`;
  const isWin = st.i2 > st.i1;
  document.getElementById("tlabel").textContent = isWin
    ? `窗口 [${p(d.times[st.i1])}, ${p(d.times[st.i2])}]`
    : `t=${p(d.times[st.i1])}`;
  document.getElementById("rstats").innerHTML =
    `${st.mode === "topo" ? "拓扑图" : "ring矩阵"} ｜ 通道 ${st.channel} ｜ ${isWin
      ? `窗口 [${p(d.times[st.i1])}, ${p(d.times[st.i2])}] 均值（单击时间条回到选点）`
      : `时刻 t=${p(d.times[st.i1])}（按住拖动时间条可选窗口）`}`;
}

function fillRingFilter(parsed) {      // 每条 ring 一个复选框，默认全选
  const host = document.getElementById("ring-filter");
  host.innerHTML = "<span>Ring：</span>";
  parsed.rings.forEach(r => {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !st.hidden.has(r);
    cb.onchange = () => {
      if (cb.checked) st.hidden.delete(r); else st.hidden.add(r);
      topoRefs = null;                 // 拓扑行需增删，强制重建
      update();
    };
    label.appendChild(cb);
    label.appendChild(document.createTextNode(r));
    host.appendChild(label);
  });
}

function fillTopoCh(parsed) {          // 拓扑模式的通道下拉
  const sel = document.getElementById("topo-ch");
  sel.innerHTML = "";
  parsed.channels.forEach(c => {
    const o = document.createElement("option");
    o.value = o.textContent = c;
    sel.appendChild(o);
  });
  sel.value = st.channel;
  sel.onchange = () => { st.channel = sel.value; update(); };
}

function applyMode() {                 // 模式切换：显隐对应控件与容器
  const topo = st.mode === "topo";
  document.getElementById("mod-sel").style.display = topo ? "none" : "";
  document.getElementById("stat-sel").style.display = topo ? "none" : "";
  document.getElementById("topo-ch").style.display = topo ? "" : "none";
  document.getElementById("topo").style.display = topo ? "" : "none";
  document.getElementById("rtable").style.display = topo ? "none" : "";
}

function render() {
  const d = DATA[st.type];
  const parsed = parseNoc(d);
  const info = document.getElementById("info");
  if (!parsed) {
    info.textContent = `类型 ${st.type} 行标不符合 R_C_CS 三段约定`;
    document.getElementById("rtable").innerHTML = "";
    document.getElementById("topo").innerHTML = "";
    return;
  }
  st.parsed = parsed;
  if (!st.channel || !parsed.channels.includes(st.channel))
    st.channel = parsed.channels[0];
  if (!(st.i1 >= 0 && st.i1 < d.times.length)) st.i1 = 0;
  if (!(st.i2 >= st.i1 && st.i2 < d.times.length)) st.i2 = st.i1;
  const desc = Object.entries(d.meta || {}).map(([k, v]) => `${k}=${v}`).join(" · ");
  info.textContent = `${desc} ｜ ${d.modules.length} 模块 × ${d.times.length} 窗口` +
    (d.downsample ? ` ｜ 时间轴已压缩 ${d.downsample}:1（合并口径=${d.dsh}）` : "");
  fillRingFilter(parsed);
  fillTopoCh(parsed);
  if (st.mode === "topo") buildTopo();
  update();
}

// 模块 -> 统计项 两级选择；统计项按「类型.通道」平铺（如 noc_left_util.Req）
const OPTIONS = [];
ALL_TYPES.forEach(t => {
  const parsed = parseNoc(DATA[t]);
  if (parsed) parsed.channels.forEach(ch => OPTIONS.push({ type: t, channel: ch }));
});
const modSel = document.getElementById("mod-sel");
const statSel = document.getElementById("stat-sel");
const fillStats = mod => {
  statSel.innerHTML = "";
  OPTIONS.forEach((o, i) => {
    if (moduleOf(o.type) !== mod) return;
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${o.type}.${o.channel}`;
    statSel.appendChild(opt);
  });
};
[...new Set(ALL_TYPES.map(moduleOf))].sort().forEach(m => {
  const o = document.createElement("option");
  o.value = o.textContent = m;
  modSel.appendChild(o);
});
st.type = OPTIONS[0].type;
st.channel = OPTIONS[0].channel;
modSel.value = moduleOf(st.type);
fillStats(modSel.value);
statSel.value = statSel.options[0].value;
const onTypeChange = () => {
  if (st.timer) { clearInterval(st.timer); st.timer = null; }
  const o = OPTIONS[+statSel.value];
  st.type = o.type; st.channel = o.channel; st.i1 = 0; st.i2 = 0;
  render();
};
modSel.onchange = () => {
  fillStats(modSel.value);
  statSel.value = statSel.options[0].value;
  onTypeChange();
};
statSel.onchange = () => onTypeChange();

document.querySelectorAll("#mode-tabs button").forEach(b => b.onclick = () => {
  if (st.mode === b.dataset.mode) return;
  st.mode = b.dataset.mode;
  document.querySelectorAll("#mode-tabs button").forEach(x =>
    x.classList.toggle("active", x === b));
  applyMode();
  render();
});

applyMode();
initTimeBar();
render();
"""


def export_report(
    store: ProfilingStore,
    out,
    types: list[str] | None = None,
    modules: list[str] | None = None,
    module_prefix: str | None = None,
    t_range=None,
    top_n: int = 20,
) -> Path:
    """生成多面板 HTML 观测器，返回文件路径。"""
    try:
        import plotly  # noqa: F401  -- 先确认包可导入，再取内联 JS
        from plotly.offline import get_plotlyjs
    except ImportError:
        raise ImportError(
            "生成报告需要 plotly：pip install plotly（或 pip install 'profmw[viz]'）"
        ) from None

    selected = types if types is not None else list(store.list_types())
    if not selected:
        raise ValueError("没有任何 profiling 类型")

    data = {}
    for ptype in selected:
        df = store.window_series(ptype, modules=modules, module_prefix=module_prefix,
                                 t_range=t_range)
        data[ptype] = {
            "meta": store.list_types().get(ptype, {}),
            "times": [float(t) for t in df.columns],
            "modules": [str(m) for m in df.index],
            "values": df.to_numpy(dtype=float).tolist(),  # NaN 输出为 JS 字面量
        }

    meta_line = " · ".join(
        "levels=" + ">".join(lv["key"] for lv in v) if k == "levels" else f"{k}={v}"
        for k, v in store.meta.items()
        if k not in ("noc_node_names", "noc_ring_die")   # 拓扑框专用映射不上页头
    ) or "（无 run meta）"  # levels 压缩为 R>C>CS，避免整段 JSON 糊在页头
    app_js = (_JS_APP
              .replace("__DATA__", json.dumps(data, ensure_ascii=False))
              .replace("__TOPN__", str(int(top_n)))
              .replace("__NODES__",
                       json.dumps(store.meta.get("noc_node_names") or {},
                                  ensure_ascii=False))
              .replace("__RING_DIE__",
                       json.dumps(store.meta.get("noc_ring_die") or {},
                                  ensure_ascii=False)))
    html = "\n".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>profmw 观测器</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<p class='meta'>{meta_line}</p>",
        "<div class='controls'>",
        "<button id=\"add-panel\">＋ 新增波形框</button>",
        "<button id=\"add-topo-panel\">＋ 新增拓扑框</button>",
        "<button id=\"add-latency-panel\">＋ 新增时延框</button>",
        "</div>",
        "<div id=\"panels\"></div>",
        "<script>", get_plotlyjs(), "</script>",
        "<script>", app_js, "</script>",
        "</body></html>",
    ])
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    return out


def _is_noc_shaped(df: pd.DataFrame) -> bool:
    """行标全部符合 R_C_CS 三段约定（下划线分隔的 3 个非空段）。"""
    for m in df.index:
        t = str(m).split("_")
        if len(t) != 3 or not all(t):
            return False
    return True


def _downsample_time(df: pd.DataFrame, k: int, how: str) -> pd.DataFrame:
    """把相邻 k 个时间窗口合并为 1 个（列标取组内窗口起点），模块/行完整保留。

    how="sum" 按窗口求和（count 类），"mean" 取均值（率/占用类）；NaN 跳过，
    全 NaN 的组保持 NaN。
    """
    vals = df.to_numpy(dtype=float)
    n_mod, n_t = vals.shape
    n_groups = -(-n_t // k)
    times = np.asarray(df.columns, dtype=float)
    out = np.empty((n_mod, n_groups), dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for gi in range(n_groups):
            lo, hi = gi * k, min((gi + 1) * k, n_t)
            seg = vals[:, lo:hi]
            if how == "sum":
                col = np.nansum(seg, axis=1)
                col[np.isnan(seg).all(axis=1)] = np.nan
            else:
                col = np.nanmean(seg, axis=1)
            out[:, gi] = col
    return pd.DataFrame(out, index=df.index,
                        columns=[float(times[gi * k]) for gi in range(n_groups)])


def export_ring_report(
    store: ProfilingStore,
    out,
    types: list[str] | None = None,
    modules: list[str] | None = None,
    module_prefix: str | None = None,
    t_range=None,
    max_points: int | None = 300_000,
) -> Path:
    """生成 ring矩阵 专用单文件 HTML（行=ring、列=CS 染色数值表）。

    纯 DOM 页面，不渲染任何图表，**无 plotly 依赖**。
    types 缺省时自动选择行标符合 R_C_CS 约定的类型；
    max_points（默认 300000，None 表示不限制）限制每类型总采样点
    （模块数×窗口数），超限时**仅压缩时间轴**——
    相邻窗口按 k:1 合并（k=ceil(总点/上限)），模块/CS 完整保留；
    合并口径按类型 unit：count -> 求和，其余 -> 均值；压缩比标注在页面上。
    """
    if types is None:
        types = [t for t in store.list_types() if _is_noc_shaped(store.get_matrix(t))]
        if not types:
            raise ValueError("没有任何行标符合 R_C_CS 约定的类型")
    data = {}
    for ptype in types:
        df = store.window_series(ptype, modules=modules, module_prefix=module_prefix,
                                 t_range=t_range)
        if not _is_noc_shaped(df):
            raise ValueError(f"类型 {ptype!r} 行标不符合 R_C_CS 三段约定")
        meta = store.list_types().get(ptype, {})
        k, how = None, None
        if max_points is not None:
            total = df.shape[0] * df.shape[1]
            if total > max_points:
                k = -(-total // max_points)
                how = "sum" if meta.get("unit") == "count" else "mean"
                df = _downsample_time(df, k, how)
        data[ptype] = {
            "meta": meta,
            "times": [float(t) for t in df.columns],
            "modules": [str(m) for m in df.index],
            "values": df.to_numpy(dtype=float).tolist(),
            "downsample": k,
            "dsh": how,
        }

    meta_line = " · ".join(
        "levels=" + ">".join(lv["key"] for lv in v) if k == "levels" else f"{k}={v}"
        for k, v in store.meta.items()
        if k not in ("noc_node_names", "noc_ring_die")   # 拓扑专用映射不上页头
    ) or "（无 run meta）"
    app_js = (_JS_RING_APP
              .replace("__DATA__", json.dumps(data, ensure_ascii=False))
              .replace("__NODES__",
                       json.dumps(store.meta.get("noc_node_names") or {},
                                  ensure_ascii=False)))
    html = "\n".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>profmw ring 观测</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<p class='meta'>{meta_line}</p>",
        "<div class='views' id='mode-tabs'>",
        "<button data-mode='topo' class='active'>拓扑图</button>",
        "<button data-mode='matrix'>矩阵表</button>",
        "</div>",
        "<div class='ring-filter' id='ring-filter'></div>",
        "<div class='p-noc' id='bar' style='display:flex'>",
        "<select id=\"mod-sel\" title=\"模块\"></select>",
        "<select id=\"stat-sel\" title=\"统计项\"></select>",
        "<select id=\"topo-ch\" title=\"通道\"></select>",
        "<div class='n-track' id='track' title=\"单击=选点；按住拖动或拖动手柄=选时间窗\">",
        "<div class='n-fill'></div><div class='n-h n-h1'></div><div class='n-h n-h2'></div></div>",
        "<span class='n-time' id='tlabel'></span>",
        "<button class='n-play' id='play'>▶ 回放</button>",
        "</div>",
        "<p class='meta' id='info'></p>",
        "<div id='topo'></div>",
        "<div id='rtable'></div>",
        "<div class='p-stats' id='rstats'></div>",
        "<script>", app_js, "</script>",
        "</body></html>",
    ])
    out = Path(out)
    out.write_text(html, encoding="utf-8")
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m profmw.report", description="从 run 目录生成 HTML 观测器")
    parser.add_argument("run", help="run 目录（或 .zip）路径")
    parser.add_argument("-o", "--out", default="report.html", help="输出 HTML 路径")
    parser.add_argument("--types", nargs="*", help="只报告指定类型")
    parser.add_argument("--prefix", help="模块名前缀过滤")
    parser.add_argument("--top", type=int, default=20, help="波形默认显示的 Top-N 模块数")
    parser.add_argument("--ring", action="store_true",
                        help="生成 ring矩阵 专用报告（纯 DOM，无需 plotly）")
    parser.add_argument("--max-points", type=int, default=300_000,
                        help="ring 报告每类型总采样点上限（默认 300000）；超限时仅压缩时间轴"
                             "（模块/CS 完整保留），口径按 unit：count 求和、其余均值")
    args = parser.parse_args(argv)
    store = ProfilingStore.load(args.run)
    if args.ring:
        out = export_ring_report(store, args.out, types=args.types,
                                  module_prefix=args.prefix,
                                  max_points=args.max_points)
    else:
        out = export_report(store, args.out, types=args.types,
                            module_prefix=args.prefix, top_n=args.top)
    print(f"报告已生成: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
