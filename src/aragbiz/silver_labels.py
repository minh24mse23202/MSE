from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from aragbiz.schemas import ComplexityLabel, QACRecord


ROUTE_ORDER = ("l1_direct", "l2_simple_rag", "l3_complex_rag", "l4_advanced_rag")
ROUTE_LABELS: Dict[str, ComplexityLabel] = {
    "l1_direct": "simple",
    "l2_simple_rag": "moderate",
    "l3_complex_rag": "complex",
    "l4_advanced_rag": "advanced",
}


@dataclass(frozen=True)
class StrategyOutcome:
    route: str
    answer_overlap: float
    faithfulness_proxy: float = 0.0
    latency_ms: float = 0.0
    estimated_cost_usd: float = 0.0
    success: Optional[bool] = None
    metadata: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HybridSilverLabeler:
    """Assign the least-complex successful route and retain complete provenance."""

    def __init__(
        self,
        strategy_runner: Callable[[QACRecord, str], StrategyOutcome],
        judge: Optional[Callable[[QACRecord, StrategyOutcome], Optional[bool]]] = None,
        *,
        judge_deployment_id: str = "",
        answer_pass_threshold: float = 0.65,
        answer_fail_threshold: float = 0.35,
        faithfulness_threshold: float = 0.60,
    ) -> None:
        self.strategy_runner = strategy_runner
        self.judge = judge
        self.judge_deployment_id = judge_deployment_id
        self.answer_pass_threshold = answer_pass_threshold
        self.answer_fail_threshold = answer_fail_threshold
        self.faithfulness_threshold = faithfulness_threshold

    def label(self, record: QACRecord, *, configuration_snapshot: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        outcomes = [self.strategy_runner(record, route) for route in ROUTE_ORDER]
        decisions: Dict[str, Optional[bool]] = {}
        judge_used = False
        for outcome in outcomes:
            decision = self._deterministic_decision(outcome)
            if decision is None and self.judge is not None:
                decision = self.judge(record, outcome)
                judge_used = judge_used or decision is not None
            decisions[outcome.route] = decision

        selected_route = next((route for route in ROUTE_ORDER if decisions.get(route) is True), "")
        if selected_route:
            label = ROUTE_LABELS[selected_route]
            provenance = "least_complex_successful_route"
        else:
            label = structural_fallback_label(record)
            provenance = "structural_fallback"

        return {
            "record_id": record.id,
            "complexity_label": label,
            "selected_route": selected_route,
            "label_provenance": provenance,
            "strategy_outcomes": [outcome.to_dict() for outcome in outcomes],
            "strategy_decisions": decisions,
            "judge_deployment_id": self.judge_deployment_id if judge_used else "",
            "judge_used": judge_used,
            "thresholds": {
                "answer_pass": self.answer_pass_threshold,
                "answer_fail": self.answer_fail_threshold,
                "faithfulness": self.faithfulness_threshold,
            },
            "configuration_snapshot": configuration_snapshot or {},
        }

    def _deterministic_decision(self, outcome: StrategyOutcome) -> Optional[bool]:
        if outcome.success is not None:
            return outcome.success
        if outcome.answer_overlap < self.answer_fail_threshold:
            return False
        if outcome.answer_overlap < self.answer_pass_threshold:
            return None
        if outcome.route != "l1_direct" and outcome.faithfulness_proxy < self.faithfulness_threshold:
            return False
        return True


def structural_fallback_label(record: QACRecord) -> ComplexityLabel:
    metadata = record.metadata or {}
    source_ids = metadata.get("article_ids") or metadata.get("document_ids") or metadata.get("source_ids") or []
    if isinstance(source_ids, str):
        source_ids = [source_ids]
    source_count = len(source_ids)
    reasoning_hops = int(metadata.get("reasoning_hops") or metadata.get("hop_count") or source_count or 0)
    connector_required = bool(metadata.get("connector_required") or metadata.get("research_required"))
    if connector_required or source_count >= 4 or reasoning_hops >= 4:
        return "advanced"
    if source_count >= 2 or reasoning_hops >= 2:
        return "complex"
    if source_count == 1 or reasoning_hops == 1:
        return "moderate"
    return "simple"
