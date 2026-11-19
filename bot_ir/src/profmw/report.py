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
const ALL_TYPES = Object.keys(DATA).sort();
// 模块分组：类型 meta 的 module 字段；缺省归"默认"组
const moduleOf = ptype => (DATA[ptype].meta && DATA[ptype].meta.module) || "默认";
const VIEWS = [["lines", "波形"], ["heatmap", "热力图"], ["stats", "统计表"]];
const TIME_VIEWS = new Set(["lines", "heatmap"]);
const panels = [];
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

function addPanel(type, view) {
  const id = "p" + panelSeq++;
  const wrap = document.createElement("div");
  wrap.className = "panel";
  wrap.innerHTML =
    `<div class="panel-head">` +
    `<select class="p-mod" title="模块"></select>` +
    `<select class="p-stat" title="统计项"></select>` +
    `<select class="p-view"></select>` +
    `<label class="p-sum" title="显示当前可见模块的逐时间点求和曲线"><input type="checkbox"> Σ 求和曲线</label>` +
    `<button class="p-close" title="移除面板">×</button>` +
    `<span class="p-info"></span></div>` +
    `<div class="p-chart" id="${id}"></div>` +
    `<div class="p-table"></div>` +
    `<div class="p-stats" id="${id}-stats"></div>`;
  document.getElementById("panels").appendChild(wrap);

  const modSel = wrap.querySelector(".p-mod");
  [...new Set(ALL_TYPES.map(moduleOf))].sort().forEach(m => {
    const o = document.createElement("option");
    o.value = o.textContent = m;
    modSel.appendChild(o);
  });
  const statSel = wrap.querySelector(".p-stat");
  const fillStats = mod => {
    statSel.innerHTML = "";
    ALL_TYPES.filter(t => moduleOf(t) === mod).forEach(t => {
      const o = document.createElement("option");
      o.value = o.textContent = t;
      statSel.appendChild(o);
    });
  };
  const initMod = moduleOf(type);
  modSel.value = initMod;
  fillStats(initMod);
  statSel.value = type;
  const vsel = wrap.querySelector(".p-view");
  VIEWS.forEach(([v, name]) => {
    const o = document.createElement("option");
    o.value = v; o.textContent = name;
    vsel.appendChild(o);
  });
  vsel.value = view;

  const panel = { id, type, view, xrange: null, selrange: null, showSum: false,
                  wired: false, el: wrap };
  panels.push(panel);
  modSel.onchange = () => {           // 切模块：统计项列表跟随刷新
    fillStats(modSel.value);
    statSel.value = statSel.options[0].value;
    panel.type = statSel.value; panel.xrange = null; panel.selrange = null;
    renderPanel(panel);
  };
  statSel.onchange = () => {
    panel.type = statSel.value; panel.xrange = null; panel.selrange = null;
    renderPanel(panel);
  };
  vsel.onchange = () => {
    panel.view = vsel.value; panel.xrange = null; panel.selrange = null;
    renderPanel(panel);
  };
  wrap.querySelector(".p-sum input").onchange = e => {
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
  const show = panels.length > 1;
  panels.forEach(pn =>
    pn.el.querySelector(".p-close").style.display = show ? "" : "none");
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

document.getElementById("add-panel").onclick = () => {
  addPanel(ALL_TYPES[panels.length % ALL_TYPES.length], "lines");
};
addPanel(ALL_TYPES[0], "lines");
"""

_JS_RING_APP = r"""
const DATA = __DATA__;
const ALL_TYPES = Object.keys(DATA).sort();
const moduleOf = t => (DATA[t].meta && DATA[t].meta.module) || "默认";
const st = { type: ALL_TYPES[0], channel: null, i1: 0, i2: 0, timer: null,
             parsed: null };

function p(v) { return isFinite(v) ? Number(v.toPrecision(6)) : "NaN"; }

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
  return { rings: [...rings].sort(), channels: [...channels].sort(), csByCh, map };
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

function fillChannels(parsed) {
  const chSel = document.getElementById("ch-sel");
  chSel.innerHTML = "";
  parsed.channels.forEach(c => {
    const o = document.createElement("option");
    o.value = o.textContent = c;
    chSel.appendChild(o);
  });
  chSel.value = st.channel;
  chSel.onchange = () => { st.channel = chSel.value; update(); };
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
  const sc = nocDerived(d);
  const csList = parsed.csByCh[st.channel] || [];
  const head = "<tr><th></th>" + csList.map(c => `<th>${c}</th>`).join("") + "</tr>";
  const rows = parsed.rings.map(r => {
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
    `ring矩阵 ｜ 通道 ${st.channel} ｜ ${isWin
      ? `窗口 [${p(d.times[st.i1])}, ${p(d.times[st.i2])}] 均值（单击时间条回到选点）`
      : `时刻 t=${p(d.times[st.i1])}（按住拖动时间条可选窗口）`} ｜ 色标 [${p(sc.min)}, ${p(sc.maxVal)}]`;
}

function render() {
  const d = DATA[st.type];
  const parsed = parseNoc(d);
  const info = document.getElementById("info");
  if (!parsed) {
    info.textContent = `类型 ${st.type} 行标不符合 R_C_CS 三段约定`;
    document.getElementById("rtable").innerHTML = "";
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
  fillChannels(parsed);
  update();
}

// 模块 -> 统计项 两级类型选择
const modSel = document.getElementById("mod-sel");
const statSel = document.getElementById("stat-sel");
const fillStats = mod => {
  statSel.innerHTML = "";
  ALL_TYPES.filter(t => moduleOf(t) === mod).forEach(t => {
    const o = document.createElement("option");
    o.value = o.textContent = t;
    statSel.appendChild(o);
  });
};
[...new Set(ALL_TYPES.map(moduleOf))].sort().forEach(m => {
  const o = document.createElement("option");
  o.value = o.textContent = m;
  modSel.appendChild(o);
});
modSel.value = moduleOf(st.type);
fillStats(modSel.value);
statSel.value = st.type;
const onTypeChange = () => {
  if (st.timer) { clearInterval(st.timer); st.timer = null; }
  st.channel = null; st.i1 = 0; st.i2 = 0;
  render();
};
modSel.onchange = () => {
  fillStats(modSel.value);
  statSel.value = statSel.options[0].value;
  st.type = statSel.value;
  onTypeChange();
};
statSel.onchange = () => { st.type = statSel.value; onTypeChange(); };

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
    ) or "（无 run meta）"  # levels 压缩为 R>C>CS，避免整段 JSON 糊在页头
    app_js = (_JS_APP
              .replace("__DATA__", json.dumps(data, ensure_ascii=False))
              .replace("__TOPN__", str(int(top_n))))
    html = "\n".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>profmw 观测器</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<p class='meta'>{meta_line}</p>",
        "<div class='controls'>",
        "<button id=\"add-panel\">＋ 新增波形框</button>",
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
    ) or "（无 run meta）"
    app_js = _JS_RING_APP.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = "\n".join([
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>profmw ring矩阵</title>",
        f"<style>{_CSS}</style></head><body>",
        f"<p class='meta'>{meta_line}</p>",
        "<div class='p-noc' id='bar' style='display:flex'>",
        "<select id=\"mod-sel\" title=\"模块\"></select>",
        "<select id=\"stat-sel\" title=\"统计项\"></select>",
        "<select id=\"ch-sel\" title=\"通道\"></select>",
        "<div class='n-track' id='track' title=\"单击=选点；按住拖动或拖动手柄=选时间窗\">",
        "<div class='n-fill'></div><div class='n-h n-h1'></div><div class='n-h n-h2'></div></div>",
        "<span class='n-time' id='tlabel'></span>",
        "<button class='n-play' id='play'>▶ 回放</button>",
        "</div>",
        "<p class='meta' id='info'></p>",
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
