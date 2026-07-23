from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

from aragbiz.answering import AdaptiveRAGAnswerService, AnswerOptions
from aragbiz.config import load_config
from aragbiz.data import load_qac_jsonl
from aragbiz.factory import build_knowledge_service, build_model_farm_service, build_model_gateway, build_sample_pipeline
from aragbiz.model_farm import ModelCallContext
from aragbiz.preprocessing import write_qac_jsonl
from aragbiz.schemas import QACRecord
from aragbiz.silver_labels import HybridSilverLabeler, ROUTE_ORDER, StrategyOutcome


ROUTE_MODES = {
    "l1_direct": "direct",
    "l2_simple_rag": "simple_rag",
    "l3_complex_rag": "complex_rag",
    "l4_advanced_rag": "advanced_rag",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute L1-L4 and build outcome-derived silver labels.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--provenance-output", required=True)
    parser.add_argument("--chat-configuration-json", default="")
    parser.add_argument("--judge-deployment-id", default="")
    parser.add_argument("--retrieval-mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    app_config = load_config()
    model_farm = build_model_farm_service(app_config)
    gateway = build_model_gateway(app_config, model_farm_service=model_farm)
    knowledge = build_knowledge_service(app_config, model_farm_service=model_farm, model_gateway=gateway)
    pipeline = build_sample_pipeline(app_config)
    answer_service = AdaptiveRAGAnswerService(
        pipeline.router,
        pipeline.generator,
        knowledge,
        bm25_weight=app_config.bm25_weight,
        dense_weight=app_config.dense_weight,
        model_farm_service=model_farm,
        model_gateway=gateway,
    )
    knowledge_base = knowledge.get_knowledge_base(args.knowledge_base_id)
    kb_configuration = knowledge_base.metadata.get("configuration", {}) if isinstance(knowledge_base.metadata, dict) else {}
    external_processing_allowed = bool(kb_configuration.get("external_processing_allowed", False))
    chat_configuration = load_json_object(args.chat_configuration_json) if args.chat_configuration_json else {}
    snapshot = {
        "knowledge_base_id": args.knowledge_base_id,
        "retrieval_mode": args.retrieval_mode,
        "top_k": args.top_k,
        "chat_configuration": chat_configuration,
        "judge_deployment_id": args.judge_deployment_id,
    }

    def run_strategy(record: QACRecord, route: str) -> StrategyOutcome:
        result = answer_service.answer(
            record.question,
            AnswerOptions(
                mode=ROUTE_MODES[route],  # type: ignore[arg-type]
                knowledge_base_id=args.knowledge_base_id if route != "l1_direct" else None,
                retrieval_mode=args.retrieval_mode,  # type: ignore[arg-type]
                top_k=args.top_k,
                chat_configuration=chat_configuration,
            ),
        )
        trace_summary = result.metadata.get("trace_summary") or {}
        return StrategyOutcome(
            route=route,
            answer_overlap=answer_overlap(record.answer, result.answer),
            faithfulness_proxy=faithfulness_proxy(result.answer, [context.document.text for context in result.contexts]),
            latency_ms=float(result.metadata.get("latency_ms") or 0.0),
            estimated_cost_usd=float(
                trace_summary.get("estimated_cost_usd")
                or result.metadata.get("actual_generator", {}).get("estimated_cost_usd")
                or 0.0
            ),
            metadata={
                "answer": result.answer,
                "route_level": result.metadata.get("route_level"),
                "context_ids": [context.document.id for context in result.contexts],
                "configured_classifier": result.metadata.get("configured_classifier", {}),
                "configured_planner": result.metadata.get("configured_planner", {}),
            },
        )

    def judge(record: QACRecord, outcome: StrategyOutcome) -> bool | None:
        if not args.judge_deployment_id:
            return None
        try:
            response = gateway.generate_sync(
                [
                    {
                        "role": "system",
                        "content": (
                            "Judge whether the candidate correctly answers the reference business-workflow question. "
                            "Return only JSON: {\"success\": true} or {\"success\": false}."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "question": record.question,
                                "reference_answer": record.answer,
                                "candidate_answer": (outcome.metadata or {}).get("answer", ""),
                                "route": outcome.route,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                args.judge_deployment_id,
                parameters={"temperature": 0, "max_tokens": 32},
                context=ModelCallContext(purpose="silver_label_judge"),
                external_processing_allowed=external_processing_allowed,
                capability="judge",
            )
            payload = json.loads(strip_json_fence(response.text))
            value = payload.get("success") if isinstance(payload, dict) else None
            return value if isinstance(value, bool) else None
        except (ValueError, RuntimeError):
            return None

    records = load_qac_jsonl(args.dataset)
    if args.limit > 0:
        records = records[: args.limit]
    labeler = HybridSilverLabeler(
        run_strategy,
        judge if args.judge_deployment_id else None,
        judge_deployment_id=args.judge_deployment_id,
    )
    provenance = [labeler.label(record, configuration_snapshot=snapshot) for record in records]
    by_id = {item["record_id"]: item for item in provenance}
    labeled = [
        QACRecord(
            record.id,
            record.question,
            record.answer,
            record.context,
            by_id[record.id]["complexity_label"],
            {**record.metadata, "complexity_labeling": by_id[record.id]},
        )
        for record in records
    ]
    write_qac_jsonl(labeled, Path(args.output))
    write_jsonl(provenance, args.provenance_output)
    print(json.dumps({"records": len(records), "routes": list(ROUTE_ORDER), "output": args.output}, indent=2))


def answer_overlap(expected: str, actual: str) -> float:
    expected_terms = set(re.findall(r"[a-z0-9]+", expected.lower()))
    actual_terms = set(re.findall(r"[a-z0-9]+", actual.lower()))
    return len(expected_terms & actual_terms) / len(expected_terms) if expected_terms else 0.0


def faithfulness_proxy(answer: str, contexts: list[str]) -> float:
    terms = set(re.findall(r"[a-z0-9]+", answer.lower()))
    corpus = " ".join(contexts).lower()
    return len({term for term in terms if term in corpus}) / len(terms) if terms else 0.0


def strip_json_fence(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def load_json_object(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Chat configuration snapshot must be a JSON object.")
    return payload


def write_jsonl(records: list[Dict[str, Any]], path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
