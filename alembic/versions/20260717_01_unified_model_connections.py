"""Separate model connections from deployments.

Revision ID: 20260717_01
Revises: 20260713_03
Create Date: 2026-07-17
"""
from alembic import op


revision = "20260717_01"
down_revision = "20260713_03"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS model_connections (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            provider TEXT NOT NULL,
            access_path TEXT NOT NULL,
            api_base TEXT NOT NULL DEFAULT '',
            credential_env_refs_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            credential_secrets_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            locality TEXT NOT NULL DEFAULT 'remote',
            enabled BOOLEAN NOT NULL DEFAULT FALSE,
            health_status TEXT NOT NULL DEFAULT 'untested',
            last_health_check TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    op.execute("ALTER TABLE model_deployments ADD COLUMN IF NOT EXISTS connection_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE model_usage_events ADD COLUMN IF NOT EXISTS connection_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE model_usage_events ADD COLUMN IF NOT EXISTS access_path TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE model_usage_events ADD COLUMN IF NOT EXISTS gateway_model TEXT NOT NULL DEFAULT ''")

    op.execute(
        """
        INSERT INTO model_connections (
            id, name, provider, access_path, api_base, credential_env_refs_json,
            credential_secrets_json, locality, enabled, health_status,
            last_health_check, last_error, metadata_json, created_at, updated_at
        ) VALUES (
            'connection-local-builtin', 'Built-in local runtime', 'local_builtin', 'local', '',
            '{}'::jsonb, '{}'::jsonb, 'local', TRUE, 'healthy', '', '',
            '{"builtin": true}'::jsonb, NOW()::text, NOW()::text
        )
        ON CONFLICT (id) DO UPDATE SET
            provider = 'local_builtin', access_path = 'local', locality = 'local',
            enabled = TRUE, health_status = 'healthy', updated_at = NOW()::text
        """
    )
    op.execute(
        """
        UPDATE model_deployments
        SET connection_id = 'connection-local-builtin'
        WHERE connection_id = ''
          AND (LOWER(provider) = 'local' OR COALESCE(metadata_json->>'builtin', 'false') = 'true')
        """
    )

    # Preserve every legacy deployment while sharing a connection for identical
    # provider/base/credential/locality tuples. PostgreSQL's built-in md5 keeps
    # the migration independent from pgcrypto.
    op.execute(
        """
        WITH legacy AS (
            SELECT
                md.*,
                CASE
                    WHEN LOWER(md.provider) = 'openrouter'
                      OR LOWER(md.api_base) LIKE '%openrouter.ai%'
                      OR LOWER(md.model) LIKE 'openrouter/%'
                      OR RIGHT(LOWER(md.model), 5) = CHR(58) || 'free' THEN 'openrouter'
                    WHEN LOWER(md.provider) IN ('gemini', 'google') THEN 'gemini'
                    WHEN LOWER(md.provider) IN ('ollama', 'ollama_chat') THEN 'ollama'
                    WHEN LOWER(md.provider) IN ('vllm', 'hosted_vllm') THEN 'vllm'
                    ELSE 'openai'
                END AS connection_provider,
                'connection-migrated-' || SUBSTR(MD5(
                    COALESCE(LOWER(md.provider), '') || '|' ||
                    COALESCE(LOWER(md.api_base), '') || '|' ||
                    COALESCE(md.credential_env_refs_json::text, '{}') || '|' ||
                    COALESCE(md.credential_secrets_json::text, '{}') || '|' ||
                    COALESCE(LOWER(md.locality), '')
                ), 1, 20) AS migrated_connection_id
            FROM model_deployments md
            WHERE md.connection_id = ''
        ), unique_connections AS (
            SELECT DISTINCT ON (migrated_connection_id)
                migrated_connection_id, name, connection_provider, api_base,
                credential_env_refs_json, credential_secrets_json, locality,
                enabled, health_status, last_health_check, last_error, created_at, updated_at
            FROM legacy
            ORDER BY migrated_connection_id, id
        )
        INSERT INTO model_connections (
            id, name, provider, access_path, api_base, credential_env_refs_json,
            credential_secrets_json, locality, enabled, health_status,
            last_health_check, last_error, metadata_json, created_at, updated_at
        )
        SELECT
            migrated_connection_id,
            name || ' connection ' || RIGHT(migrated_connection_id, 6),
            connection_provider,
            CASE WHEN connection_provider = 'openrouter' THEN 'experimentation'
                 WHEN connection_provider IN ('ollama', 'vllm') THEN 'local'
                 ELSE 'production' END,
            api_base,
            credential_env_refs_json,
            credential_secrets_json,
            CASE WHEN connection_provider IN ('ollama', 'vllm') THEN 'local' ELSE locality END,
            enabled,
            health_status,
            last_health_check,
            last_error,
            '{"migrated": true}'::jsonb,
            created_at,
            updated_at
        FROM unique_connections
        ON CONFLICT (id) DO NOTHING
        """
    )
    op.execute(
        """
        WITH legacy AS (
            SELECT
                id,
                'connection-migrated-' || SUBSTR(MD5(
                    COALESCE(LOWER(provider), '') || '|' ||
                    COALESCE(LOWER(api_base), '') || '|' ||
                    COALESCE(credential_env_refs_json::text, '{}') || '|' ||
                    COALESCE(credential_secrets_json::text, '{}') || '|' ||
                    COALESCE(LOWER(locality), '')
                ), 1, 20) AS migrated_connection_id
            FROM model_deployments
            WHERE connection_id = ''
        )
        UPDATE model_deployments md
        SET connection_id = legacy.migrated_connection_id
        FROM legacy
        WHERE md.id = legacy.id
        """
    )
    op.execute(
        """
        UPDATE model_usage_events usage
        SET connection_id = deployment.connection_id,
            access_path = connection.access_path,
            gateway_model = CASE
                WHEN connection.provider = 'openrouter' AND deployment.model NOT LIKE 'openrouter/%'
                    THEN 'openrouter/' || deployment.model
                WHEN connection.provider = 'gemini' AND deployment.model NOT LIKE 'gemini/%'
                    THEN 'gemini/' || deployment.model
                WHEN connection.provider = 'ollama' AND deployment.model NOT LIKE 'ollama_chat/%'
                    THEN 'ollama_chat/' || deployment.model
                WHEN connection.provider = 'vllm' AND deployment.model NOT LIKE 'hosted_vllm/%'
                    THEN 'hosted_vllm/' || deployment.model
                ELSE deployment.model
            END
        FROM model_deployments deployment
        JOIN model_connections connection ON connection.id = deployment.connection_id
        WHERE usage.deployment_id = deployment.id
          AND usage.connection_id = ''
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_connections_provider ON model_connections(provider, enabled)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_deployments_connection ON model_deployments(connection_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_usage_connection ON model_usage_events(connection_id, created_at DESC)")
    op.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'fk_model_deployments_connection'
            ) THEN
                ALTER TABLE model_deployments
                ADD CONSTRAINT fk_model_deployments_connection
                FOREIGN KEY (connection_id) REFERENCES model_connections(id) ON DELETE RESTRICT;
            END IF;
        END $$
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE model_deployments DROP CONSTRAINT IF EXISTS fk_model_deployments_connection")
    op.execute("DROP INDEX IF EXISTS idx_model_usage_connection")
    op.execute("DROP INDEX IF EXISTS idx_model_deployments_connection")
    op.execute("DROP INDEX IF EXISTS idx_model_connections_provider")
    op.execute("ALTER TABLE model_usage_events DROP COLUMN IF EXISTS gateway_model")
    op.execute("ALTER TABLE model_usage_events DROP COLUMN IF EXISTS access_path")
    op.execute("ALTER TABLE model_usage_events DROP COLUMN IF EXISTS connection_id")
    op.execute("ALTER TABLE model_deployments DROP COLUMN IF EXISTS connection_id")
    op.execute("DROP TABLE IF EXISTS model_connections")
