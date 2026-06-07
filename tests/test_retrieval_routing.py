from aragbiz.classifier import HeuristicQueryClassifier
from aragbiz.data import load_qac_jsonl, records_to_documents
from aragbiz.retrieval import InMemoryHybridRetriever
from aragbiz.routing import AdaptiveRouter


def test_classifier_returns_valid_labels():
    classifier = HeuristicQueryClassifier()
    assert classifier.predict("How do I request leave?") == "simple"
    assert classifier.predict("What happens if my manager rejects leave?") in {"moderate", "complex"}
    assert classifier.predict("How should HR process onboarding when IT access depends on signed documents?") == "complex"


def test_retriever_returns_ranked_contexts():
    records = load_qac_jsonl("data/sample/business_workflows.jsonl")
    retriever = InMemoryHybridRetriever(records_to_documents(records))
    contexts = retriever.search("invoice mismatch after goods are received", top_k=3, mode="hybrid")
    assert len(contexts) == 3
    assert [context.rank for context in contexts] == [1, 2, 3]
    assert contexts[0].document.id == "wf-003"


def test_router_maps_labels_to_strategies():
    router = AdaptiveRouter(HeuristicQueryClassifier())
    simple = router.route("How do I request leave?")
    complex_route = router.route("How should HR process onboarding when IT access depends on signed documents?")
    assert simple.retrieval_mode == "bm25"
    assert simple.top_k == 2
    assert complex_route.retrieval_mode == "hybrid"
    assert complex_route.multi_step is True

