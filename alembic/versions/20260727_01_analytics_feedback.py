"""Add analytics attribution and structured feedback.

Revision ID: 20260727_01
Revises: 20260724_01
Create Date: 2026-07-27
"""
from alembic import op


revision = "20260727_01"
down_revision = "20260724_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE IF EXISTS chat_conversations ADD COLUMN IF NOT EXISTS owner_user_id TEXT")
    op.execute("ALTER TABLE IF EXISTS chat_messages ADD COLUMN IF NOT EXISTS user_id TEXT")
    op.execute("ALTER TABLE IF EXISTS model_usage_events ADD COLUMN IF NOT EXISTS chat_configuration_id TEXT")

    op.execute(
        """
        UPDATE model_usage_events usage
        SET chat_configuration_id = evaluation.chat_configuration_id
        FROM evaluation_runs evaluation
        WHERE usage.evaluation_run_id = evaluation.id
          AND COALESCE(usage.chat_configuration_id, '') = ''
          AND COALESCE(evaluation.chat_configuration_id, '') <> ''
        """
    )
    op.execute(
        """
        UPDATE model_usage_events usage
        SET chat_configuration_id = conversation.chat_configuration_id
        FROM chat_conversations conversation
        WHERE usage.conversation_id = conversation.id
          AND COALESCE(usage.chat_configuration_id, '') = ''
          AND COALESCE(conversation.chat_configuration_id, '') <> ''
        """
    )

    op.execute(
        """
        WITH conversation_users AS (
            SELECT conversation_id, MIN(user_id) AS user_id
            FROM model_usage_events
            WHERE COALESCE(conversation_id, '') <> ''
              AND COALESCE(user_id, '') <> ''
            GROUP BY conversation_id
            HAVING COUNT(DISTINCT user_id) = 1
        )
        UPDATE chat_conversations conversation
        SET owner_user_id = users.user_id
        FROM conversation_users users
        WHERE conversation.id = users.conversation_id
          AND COALESCE(conversation.owner_user_id, '') = ''
        """
    )
    op.execute(
        """
        UPDATE chat_messages message
        SET user_id = conversation.owner_user_id
        FROM chat_conversations conversation
        WHERE message.conversation_id = conversation.id
          AND message.role = 'user'
          AND COALESCE(message.user_id, '') = ''
          AND COALESCE(conversation.owner_user_id, '') <> ''
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_feedback (
            id TEXT PRIMARY KEY,
            user_id TEXT,
            conversation_id TEXT,
            user_message_id TEXT,
            assistant_message_id TEXT NOT NULL,
            message_version_id TEXT,
            version_number INTEGER NOT NULL DEFAULT 1,
            knowledge_base_id TEXT,
            chat_configuration_id TEXT,
            trace_id TEXT,
            rating TEXT NOT NULL CHECK (rating IN ('up', 'down')),
            comment TEXT NOT NULL DEFAULT '',
            question_snapshot TEXT NOT NULL DEFAULT '',
            answer_snapshot TEXT NOT NULL DEFAULT '',
            knowledge_base_name_snapshot TEXT NOT NULL DEFAULT '',
            configuration_name_snapshot TEXT NOT NULL DEFAULT '',
            configuration_code_snapshot TEXT NOT NULL DEFAULT '',
            user_name_snapshot TEXT NOT NULL DEFAULT '',
            user_email_snapshot TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (user_id, assistant_message_id, version_number)
        )
        """
    )

    op.execute("CREATE INDEX IF NOT EXISTS idx_model_usage_created_at ON model_usage_events(created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_usage_kb_created ON model_usage_events(knowledge_base_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_usage_config_created ON model_usage_events(chat_configuration_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_usage_purpose_status ON model_usage_events(purpose, status, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_conversations_owner ON chat_conversations(owner_user_id, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_activity ON chat_messages(created_at DESC, role, status)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_user ON chat_messages(user_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_updated ON chat_feedback(updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_rating ON chat_feedback(rating, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_kb ON chat_feedback(knowledge_base_id, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_config ON chat_feedback(chat_configuration_id, updated_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_chat_feedback_user ON chat_feedback(user_id, updated_at DESC)")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS chat_feedback")
    op.execute("ALTER TABLE IF EXISTS model_usage_events DROP COLUMN IF EXISTS chat_configuration_id")
    op.execute("ALTER TABLE IF EXISTS chat_messages DROP COLUMN IF EXISTS user_id")
    op.execute("ALTER TABLE IF EXISTS chat_conversations DROP COLUMN IF EXISTS owner_user_id")
