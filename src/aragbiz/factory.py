from __future__ import annotations

from pathlib import Path
from typing import Optional

from aragbiz.classifier import HeuristicQueryClassifier, NaiveBayesQueryClassifier, QueryClassifier
from aragbiz.config import AppConfig, load_config
from aragbiz.data import load_documents_jsonl, load_qac_jsonl, records_to_documents
from aragbiz.generation import ExtractiveGenerator
from aragbiz.pipeline import RAGPipeline
from aragbiz.retrieval import InMemoryHybridRetriever
from aragbiz.routing import AdaptiveRouter, RouterConfig


def build_sample_pipeline(config: Optional[AppConfig] = None) -> RAGPipeline:
    config = config or load_config()
    dataset_path = _existing_path(config.sample_dataset, config.fallback_sample_dataset)
    records = load_qac_jsonl(dataset_path)
    if Path(config.kb_corpus).exists():
        documents = load_documents_jsonl(config.kb_corpus)
    else:
        documents = records_to_documents(records)
    classifier = build_query_classifier(config)
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


def existing_dataset_path(config: AppConfig) -> str:
    return _existing_path(config.sample_dataset, config.fallback_sample_dataset)


def build_query_classifier(config: AppConfig) -> QueryClassifier:
    model_path = Path(config.classifier_model_path)
    if config.use_trained_classifier and model_path.exists():
        return NaiveBayesQueryClassifier.load(model_path)
    return HeuristicQueryClassifier()


def _existing_path(primary: str, fallback: str) -> str:
    if Path(primary).exists():
        return primary
    return fallback
