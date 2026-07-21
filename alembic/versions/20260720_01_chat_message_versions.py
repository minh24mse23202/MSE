"""Add durable assistant message versions.

Revision ID: 20260720_01
Revises: 20260717_01
Create Date: 2026-07-20
"""
from alembic import op


revision = "20260720_01"
down_revision = "20260717_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_message_versions (
            id TEXT PRIMARY KEY,
            message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            version_number INTEGER NOT NULL,
            content TEXT NOT NULL DEFAULT '',
            contexts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            status TEXT NOT NULL DEFAULT 'completed',
            request_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (message_id, version_number)
        )
        """
    )
    op.execute(
        """
        INSERT INTO chat_message_versions (
            id, message_id, version_number, content, contexts_json, metadata_json,
            status, request_id, created_at, updated_at
        )
        SELECT
            'msgver-backfill-' || message.id,
            message.id,
            1,
            message.content,
            message.contexts_json,
            message.metadata_json,
            message.status,
            message.request_id,
            message.created_at,
            COALESCE(NULLIF(message.updated_at, ''), message.created_at)
        FROM chat_messages message
        WHERE message.role = 'assistant'
          AND NOT EXISTS (
              SELECT 1
              FROM chat_message_versions version
              WHERE version.message_id = message.id
          )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_message_versions_message "
        "ON chat_message_versions(message_id, version_number)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_message_versions_request "
        "ON chat_message_versions(request_id)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chat_message_versions_request")
    op.execute("DROP INDEX IF EXISTS idx_chat_message_versions_message")
    op.execute("DROP TABLE IF EXISTS chat_message_versions")
