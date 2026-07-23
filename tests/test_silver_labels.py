from aragbiz.schemas import QACRecord
from aragbiz.silver_labels import HybridSilverLabeler, StrategyOutcome, structural_fallback_label


def _record(metadata=None):
    return QACRecord("q1", "Question", "Answer", "Context", "moderate", metadata or {})


def test_silver_label_uses_least_complex_successful_route():
    values = {
        "l1_direct": (0.2, 0.0),
        "l2_simple_rag": (0.7, 0.7),
        "l3_complex_rag": (0.9, 0.9),
        "l4_advanced_rag": (0.95, 0.95),
    }
    labeler = HybridSilverLabeler(
        lambda record, route: StrategyOutcome(route, values[route][0], values[route][1])
    )

    result = labeler.label(_record({"article_ids": ["one"]}))

    assert result["complexity_label"] == "moderate"
    assert result["selected_route"] == "l2_simple_rag"
    assert result["label_provenance"] == "least_complex_successful_route"


def test_structural_fallback_reserves_advanced_for_long_or_connector_work():
    assert structural_fallback_label(_record({"article_ids": ["one"]})) == "moderate"
    assert structural_fallback_label(_record({"article_ids": ["a", "b", "c", "d"]})) == "advanced"
    assert structural_fallback_label(_record({"connector_required": True})) == "advanced"
