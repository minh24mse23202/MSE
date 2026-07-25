from __future__ import annotations

import argparse
import json
from pathlib import Path

from aragbiz.chat import JsonChatRepository, ChatService
from aragbiz.config import load_config
from aragbiz.evaluation import EvaluationRunConfig
from aragbiz.factory import build_evaluation_service, build_knowledge_service, build_sample_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run and persist one RAG configuration evaluation snapshot.")
    parser.add_argument("--knowledge-base-id", required=True, help="Knowledge base ID to evaluate against.")
    parser.add_argument("--chat-configuration-id", default=None, help="Optional saved chat configuration ID.")
    parser.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", default="docs/evaluation/results", help="Directory for JSON result snapshots.")
    args = parser.parse_args()

    config = load_config()
    knowledge_service = build_knowledge_service(config)
    evaluation_service = build_evaluation_service(
        config,
        knowledge_service=knowledge_service,
        pipeline=build_sample_pipeline(config),
    )
    chat_service = ChatService(JsonChatRepository(config.chat_json_store))
    chat_configuration_id = args.chat_configuration_id
    if chat_configuration_id:
        chat_configuration = _configuration_snapshot(chat_service.get_configuration(chat_configuration_id))
    else:
        default_record = chat_service.default_configuration()
        chat_configuration_id = default_record.id
        chat_configuration = _configuration_snapshot(default_record)

    run = evaluation_service.run(
        EvaluationRunConfig(
            knowledge_base_id=args.knowledge_base_id,
            chat_configuration_id=chat_configuration_id,
            chat_configuration=chat_configuration,
            retrieval_mode=args.retrieval_mode,
            top_k=args.top_k,
            limit=args.limit,
        )
    )
    cases = evaluation_service.list_cases(run.id)
    output = {
        "run": _run_payload(run),
        "cases": [_case_payload(case) for case in cases],
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run.id}.json"
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "run_id": run.id,
                "metrics": run.metrics,
                "ragxplain": run.metadata.get("ragxplain", {}),
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _configuration_snapshot(record):
    return {
        "id": record.id,
        "name": record.name,
        "description": record.description,
        "generator_provider": record.generator_provider,
        "generator_model": record.generator_model,
        "response_structure": record.response_structure,
        "tone": record.tone,
        "humor_level": record.humor_level,
        "system_prompt": record.system_prompt,
        "predefined_prompt": record.predefined_prompt,
        "metadata": record.metadata,
    }


def _run_payload(record):
    return {
        "id": record.id,
        "name": record.name,
        "dataset_name": record.dataset_name,
        "status": record.status,
        "knowledge_base_id": record.knowledge_base_id,
        "knowledge_base_name": record.knowledge_base_name,
        "chat_configuration_id": record.chat_configuration_id,
        "retrieval_mode": record.retrieval_mode,
        "top_k": record.top_k,
        "limit": record.limit,
        "metrics": record.metrics,
        "route_distribution": record.route_distribution,
        "metadata": record.metadata,
        "error": record.error,
        "created_at": record.created_at,
        "finished_at": record.finished_at,
    }


def _case_payload(record):
    return {
        "id": record.id,
        "run_id": record.run_id,
        "record_id": record.record_id,
        "question": record.question,
        "expected_answer": record.expected_answer,
        "complexity_label": record.complexity_label,
        "answer": record.adaptive_answer,
        "contexts": record.adaptive_contexts,
        "answer_metadata": record.adaptive_metadata,
        "metrics": record.metrics,
        "created_at": record.created_at,
    }


if __name__ == "__main__":
    main()
