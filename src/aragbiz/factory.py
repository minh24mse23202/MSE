from __future__ import annotations

from typing import Optional

from aragbiz.classifier import HeuristicQueryClassifier
from aragbiz.config import AppConfig, load_config
from aragbiz.data import load_qac_jsonl, records_to_documents
from aragbiz.generation import ExtractiveGenerator
from aragbiz.pipeline import RAGPipeline
from aragbiz.retrieval import InMemoryHybridRetriever
from aragbiz.routing import AdaptiveRouter, RouterConfig


def build_sample_pipeline(config: Optional[AppConfig] = None) -> RAGPipeline:
    config = config or load_config()
    records = load_qac_jsonl(config.sample_dataset)
    documents = records_to_documents(records)
    classifier = HeuristicQueryClassifier()
    router = AdaptiveRouter(
        classifier,
        RouterConfig(
            simple_top_k=config.simple_top_k,
            moderate_top_k=config.moderate_top_k,
            complex_top_k=config.complex_top_k,
        ),
    )
    retriever = InMemoryHybridRetriever(
        documents,
        bm25_weight=config.bm25_weight,
        dense_weight=config.dense_weight,
    )
    generator = ExtractiveGenerator(max_context_chars=config.max_context_chars)
    return RAGPipeline(router=router, retriever=retriever, generator=generator)
