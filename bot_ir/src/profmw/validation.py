"""校验：对入库数据的健康检查。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TypeIssue:
    """单个类型的一条校验结果。level ∈ ok/warn/error。"""

    ptype: str
    level: str
    message: str


@dataclass
class ValidationReport:
    issues: list[TypeIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """无 error 且无 warn 时为 True。"""
        return all(i.level == "ok" for i in self.issues)

    def __str__(self) -> str:
        if not self.issues:
            return "校验通过：无数据。"
        lines = [f"[{i.level.upper():5}] {i.ptype}: {i.message}" for i in self.issues]
        return "\n".join(lines)


def validate_store(store, nan_ratio_warn: float = 0.5) -> ValidationReport:
    """逐类型检查：空矩阵(error)、inf(error)、NaN 占比超阈值(warn)。"""
    report = ValidationReport()
    for ptype, df in store._matrices.items():
        if df.shape[0] == 0 or df.shape[1] == 0:
            report.issues.append(TypeIssue(ptype, "error", f"空矩阵 {df.shape}"))
            continue
        arr = df.to_numpy(dtype=float)
        msgs = []
        if np.isinf(arr).any():
            msgs.append((ptype, "error", f"含 {int(np.isinf(arr).sum())} 个非有限值(inf)"))
        nan_ratio = float(np.isnan(arr).mean())
        if nan_ratio > nan_ratio_warn:
            msgs.append((ptype, "warn", f"NaN 占比 {nan_ratio:.1%} 超过阈值 {nan_ratio_warn:.0%}"))
        if msgs:
            report.issues.extend(TypeIssue(*m) for m in msgs)
        else:
            report.issues.append(TypeIssue(ptype, "ok", "无问题"))
    return report
