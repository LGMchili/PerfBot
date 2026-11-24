"""NoC ring 数据接入：noc_raw 目录 -> ProfilingStore。

层级约定（run meta 的 levels）：ring -> channel -> cs，行标 =
{R}{ring:02d}_{通道}_{CS}{cs:02d}，通道 ∈ Req/Snp/Rsp/Dat，如 R03_Rsp_CS13。
ring/CS 零填充保证字典序 = 物理序，前缀过滤即层级聚合
（"R00_" = 整条 ring，"R00_Rsp_" = 单通道）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

DEFAULT_LEVELS = [
    {"key": "R", "name": "ring", "width": 2},
    {"key": "Ch", "name": "channel", "values": ["Req", "Snp", "Rsp", "Dat"]},
    {"key": "CS", "name": "cs", "width": 2},
]


def _label_pattern(levels) -> re.Pattern:
    """由 levels 生成行标合法性的正则，如 R\\d{2}_(Req|Snp|Rsp|Dat)_CS\\d{2}。

    层级可声明 values（枚举通道名）或 width（零填充数值）。
    """
    tokens = []
    for lv in levels:
        if "values" in lv:
            tokens.append("(?:" + "|".join(map(re.escape, lv["values"])) + ")")
        else:
            tokens.append(f"{lv['key']}\\d{{{lv['width']}}}")
    return re.compile("^" + "_".join(tokens) + "$")


def add_noc_raw(store, raw_dir, levels: list | None = None) -> None:
    """从 noc_raw 目录接入：manifest.json 声明各指标 meta，每指标一张矩阵 CSV。"""
    levels = DEFAULT_LEVELS if levels is None else levels
    raw_dir = Path(raw_dir)
    manifest_file = raw_dir / "manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(f"缺少 manifest: {manifest_file}")
    if "levels" in store.meta and store.meta["levels"] != levels:
        raise ValueError(f"store 已有不同的 levels 声明: {store.meta['levels']}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    pattern = _label_pattern(levels)
    for ptype, meta in manifest.items():
        csv = raw_dir / f"{ptype}.csv"
        if not csv.exists():
            raise FileNotFoundError(f"manifest 声明的类型 {ptype!r} 缺少数据文件: {csv}")
        df = pd.read_csv(csv, index_col=0)
        bad = [m for m in df.index if not pattern.match(str(m))]
        if bad:
            raise ValueError(
                f"{ptype!r} 的行标不符合层级约定 {levels}，如: {bad[:3]}")
        store.add_matrix(ptype, df, meta=meta)
    store.meta.setdefault("levels", levels)
    names_file = raw_dir / "node_names.json"   # 可选：CS 槽位的硬件 IP 名
    if names_file.exists():
        store.meta["noc_node_names"] = json.loads(
            names_file.read_text(encoding="utf-8"))
    die_file = raw_dir / "ring_die.json"       # 可选：ring 所属 die
    if die_file.exists():
        store.meta["noc_ring_die"] = json.loads(
            die_file.read_text(encoding="utf-8"))
