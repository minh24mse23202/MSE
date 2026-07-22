"""Add full-fidelity RAG trace metadata.

Revision ID: 20260721_01
Revises: 20260720_01
Create Date: 2026-07-21
"""
from alembic import op


revision = "20260721_01"
down_revision = "20260720_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_traces (
            id TEXT PRIMARY KEY,
            request_id TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '',
            message_id TEXT NOT NULL DEFAULT '',
            message_version_id TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'running',
            route_level TEXT NOT NULL DEFAULT '',
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT '',
            duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
            warning_count INTEGER NOT NULL DEFAULT 0,
            span_count INTEGER NOT NULL DEFAULT 0,
            artifact_path TEXT NOT NULL DEFAULT '',
            artifact_sha256 TEXT NOT NULL DEFAULT '',
            artifact_size BIGINT NOT NULL DEFAULT 0,
            summary_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            expires_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_traces_request ON rag_traces(request_id)")
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_rag_traces_message "
        "ON rag_traces(message_id, message_version_id, updated_at DESC)"
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_rag_traces_expires ON rag_traces(expires_at)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_rag_traces_expires")
    op.execute("DROP INDEX IF EXISTS idx_rag_traces_message")
    op.execute("DROP INDEX IF EXISTS idx_rag_traces_request")
    op.execute("DROP TABLE IF EXISTS rag_traces")
