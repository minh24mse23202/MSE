"""Add durable chat message states.

Revision ID: 20260713_01
Revises: 20260712_01
Create Date: 2026-07-13
"""
from alembic import op


revision = "20260713_01"
down_revision = "20260712_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE IF EXISTS chat_messages ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'completed'")
    op.execute("ALTER TABLE IF EXISTS chat_messages ADD COLUMN IF NOT EXISTS request_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE IF EXISTS chat_messages ADD COLUMN IF NOT EXISTS updated_at TEXT NOT NULL DEFAULT ''")
    op.execute("UPDATE chat_messages SET status = 'completed' WHERE status = ''")
    op.execute("UPDATE chat_messages SET updated_at = created_at WHERE updated_at = ''")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_request_id ON chat_messages(request_id)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_chat_messages_request_id")
    op.execute("ALTER TABLE IF EXISTS chat_messages DROP COLUMN IF EXISTS updated_at")
    op.execute("ALTER TABLE IF EXISTS chat_messages DROP COLUMN IF EXISTS request_id")
    op.execute("ALTER TABLE IF EXISTS chat_messages DROP COLUMN IF EXISTS status")
