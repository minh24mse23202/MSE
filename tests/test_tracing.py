from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from aragbiz.tracing import FileTraceRepository, TraceArtifactStore, TraceService


def _service(tmp_path, *, max_bytes: int = 10485760) -> TraceService:
    return TraceService(
        FileTraceRepository(str(tmp_path)),
        TraceArtifactStore(str(tmp_path)),
        retention_days=30,
        max_bytes=max_bytes,
    )


def test_trace_round_trips_utf8_and_hierarchy(tmp_path) -> None:
    service = _service(tmp_path)
    recorder = service.create_recorder("request-unicode")
    span = recorder.begin_span("Retrieval", "retrieval", input_payload={"query": "Quy trình phê duyệt là gì?"})
    recorder.finish_span(span, output_payload={"answer": "Đã phê duyệt ✓"})
    recorder.finalize("completed", route_level="l2_simple_rag")

    report = service.get_report(recorder.trace_id)
    assert report["schema_version"] == "1.0"
    assert report["spans"][1]["parent_span_id"] == report["spans"][0]["span_id"]
    assert report["spans"][1]["output"]["answer"] == "Đã phê duyệt ✓"
    json.loads(json.dumps(report, ensure_ascii=False))


def test_trace_redacts_secrets_and_numeric_vectors(tmp_path) -> None:
    service = _service(tmp_path)
    recorder = service.create_recorder("request-redaction")
    span = recorder.begin_span(
        "Provider request",
        "generation",
        input_payload={
            "authorization": "Bearer secret",
            "api_key": "sk-secret",
            "embedding_vector": [0.1, 0.2, 0.3],
            "prompt": "safe prompt",
        },
    )
    recorder.finish_span(span, output_payload={"content": "safe answer"})
    recorder.finalize("completed")

    serialized = json.dumps(service.get_report(recorder.trace_id), ensure_ascii=False)
    assert "sk-secret" not in serialized
    assert "Bearer secret" not in serialized
    assert '"dimension": 3' in serialized
    assert "0.1" not in serialized


def test_trace_truncation_remains_valid_json_under_limit(tmp_path) -> None:
    service = _service(tmp_path, max_bytes=65536)
    recorder = service.create_recorder("request-large")
    for index in range(20):
        span = recorder.begin_span(f"Large span {index}", "prompt", input_payload={"prompt": "å" * 20000})
        recorder.finish_span(span, output_payload={"response": "答" * 20000})
    recorder.finalize("completed")

    record = service.get_record(recorder.trace_id)
    report = service.get_report(recorder.trace_id)
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert len(encoded) <= 65536
    assert report["truncation"]["truncated"] is True
    assert record.artifact_size > 0


def test_expired_trace_is_purged(tmp_path) -> None:
    service = _service(tmp_path)
    recorder = service.create_recorder("request-expired")
    recorder.report["expires_at"] = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    recorder.finalize("completed")

    assert service.purge_expired() == 1
    assert not list((tmp_path / "artifacts").rglob(f"{recorder.trace_id}.json.gz"))
