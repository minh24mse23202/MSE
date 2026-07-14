"""Model Farm, durable jobs, and versioned knowledge indexes.

Revision ID: 20260712_01
Revises:
Create Date: 2026-07-12
"""
import sqlalchemy as sa
from alembic import op


revision = "20260712_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS active_index_version_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE IF EXISTS knowledge_bases ADD COLUMN IF NOT EXISTS pending_index_version_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS index_version_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE IF EXISTS chunks ADD COLUMN IF NOT EXISTS parent_chunk_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE IF EXISTS chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_deployment_id TEXT NOT NULL DEFAULT ''")
    op.execute("ALTER TABLE IF EXISTS chunk_embeddings ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER NOT NULL DEFAULT 0")
    op.execute("ALTER TABLE IF EXISTS chunk_embeddings ALTER COLUMN embedding TYPE vector USING embedding::vector")
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS knowledge_index_versions (
            id TEXT PRIMARY KEY,
            knowledge_base_id TEXT NOT NULL REFERENCES knowledge_bases(id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            chunking_configuration_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            embedding_deployment_id TEXT NOT NULL DEFAULT '',
            embedding_provider TEXT NOT NULL DEFAULT '',
            embedding_model TEXT NOT NULL DEFAULT '',
            embedding_dimension INTEGER NOT NULL DEFAULT 0,
            document_count INTEGER NOT NULL DEFAULT 0,
            chunk_count INTEGER NOT NULL DEFAULT 0,
            error TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            activated_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS model_deployments (
            id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, provider TEXT NOT NULL, model TEXT NOT NULL,
            capabilities_json JSONB NOT NULL DEFAULT '[]'::jsonb, api_base TEXT NOT NULL DEFAULT '',
            credential_env_refs_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            default_parameters_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            limits_json JSONB NOT NULL DEFAULT '{}'::jsonb, pricing_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            monthly_budget_usd DOUBLE PRECISION NOT NULL DEFAULT 0, hard_budget BOOLEAN NOT NULL DEFAULT TRUE,
            locality TEXT NOT NULL DEFAULT 'remote', enabled BOOLEAN NOT NULL DEFAULT FALSE,
            health_status TEXT NOT NULL DEFAULT 'untested', last_health_check TEXT NOT NULL DEFAULT '',
            last_error TEXT NOT NULL DEFAULT '', metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS model_usage_events (
            id TEXT PRIMARY KEY, deployment_id TEXT NOT NULL REFERENCES model_deployments(id) ON DELETE RESTRICT,
            provider TEXT NOT NULL, model TEXT NOT NULL, capability TEXT NOT NULL, purpose TEXT NOT NULL,
            status TEXT NOT NULL, request_id TEXT NOT NULL DEFAULT '', user_id TEXT NOT NULL DEFAULT '',
            conversation_id TEXT NOT NULL DEFAULT '', knowledge_base_id TEXT NOT NULL DEFAULT '',
            evaluation_run_id TEXT NOT NULL DEFAULT '', input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0, total_tokens INTEGER NOT NULL DEFAULT 0,
            latency_ms DOUBLE PRECISION NOT NULL DEFAULT 0, estimated_cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
            fallback_index INTEGER NOT NULL DEFAULT 0, error_code TEXT NOT NULL DEFAULT '', error TEXT NOT NULL DEFAULT '',
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb, created_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL,
            first_name TEXT NOT NULL DEFAULT '', last_name TEXT NOT NULL DEFAULT '', role TEXT NOT NULL DEFAULT 'user',
            active BOOLEAN NOT NULL DEFAULT TRUE, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS background_jobs (
            id TEXT PRIMARY KEY, job_type TEXT NOT NULL, status TEXT NOT NULL,
            payload_json JSONB NOT NULL DEFAULT '{}'::jsonb, progress_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            result_json JSONB NOT NULL DEFAULT '{}'::jsonb, error TEXT NOT NULL DEFAULT '', attempts INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3, idempotency_key TEXT NOT NULL DEFAULT '', worker_id TEXT NOT NULL DEFAULT '',
            lease_until TEXT NOT NULL DEFAULT '', available_at TEXT NOT NULL, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, finished_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    _seed_local_deployments()
    op.execute(
        """
        INSERT INTO knowledge_index_versions (
            id, knowledge_base_id, status, chunking_configuration_json, embedding_deployment_id,
            embedding_provider, embedding_model, embedding_dimension, document_count, chunk_count,
            created_at, activated_at
        )
        SELECT 'index-legacy-' || kb.id, kb.id, 'active',
               COALESCE(kb.metadata_json->'configuration', '{}'::jsonb),
               CASE WHEN kb.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
                    THEN 'model-local-minilm-384' ELSE 'model-local-hash-384' END,
               'Local', kb.embedding_model, COALESCE(MAX(vector_dims(ce.embedding)), 384),
               COUNT(DISTINCT c.document_id), COUNT(DISTINCT c.id), kb.created_at, kb.updated_at
        FROM knowledge_bases kb
        JOIN chunks c ON c.knowledge_base_id = kb.id
        LEFT JOIN chunk_embeddings ce ON ce.chunk_id = c.id
        WHERE kb.active_index_version_id = ''
        GROUP BY kb.id
        ON CONFLICT (id) DO NOTHING
        """
    )
    op.execute("UPDATE chunks SET index_version_id = 'index-legacy-' || knowledge_base_id WHERE index_version_id = ''")
    op.execute("UPDATE knowledge_bases SET active_index_version_id = 'index-legacy-' || id WHERE active_index_version_id = '' AND EXISTS (SELECT 1 FROM chunks WHERE chunks.knowledge_base_id = knowledge_bases.id)")
    op.execute("UPDATE chunk_embeddings SET embedding_dimension = vector_dims(embedding) WHERE embedding_dimension = 0")
    op.execute(
        """
        UPDATE chunk_embeddings ce
        SET embedding_deployment_id = CASE WHEN ce.embedding_model = 'sentence-transformers/all-MiniLM-L6-v2'
                                           THEN 'model-local-minilm-384' ELSE 'model-local-hash-384' END
        WHERE ce.embedding_deployment_id = ''
        """
    )
    op.execute("CREATE INDEX IF NOT EXISTS idx_chunks_active_version ON chunks(knowledge_base_id, index_version_id)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_index_versions_kb ON knowledge_index_versions(knowledge_base_id, created_at DESC)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_model_usage_created_at ON model_usage_events(created_at DESC)")
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_background_jobs_idempotency ON background_jobs(idempotency_key) WHERE idempotency_key <> ''")


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS background_jobs")
    op.execute("DROP TABLE IF EXISTS model_usage_events")
    op.execute("DROP TABLE IF EXISTS model_deployments")
    op.execute("DROP TABLE IF EXISTS users")
    op.execute("DROP TABLE IF EXISTS knowledge_index_versions")


def _seed_local_deployments() -> None:
    rows = [
        ("model-local-extractive", "Local Extractive", "extractive", '["generation","judge"]', '{"context_window":3200,"max_output_tokens":900}'),
        ("model-local-flan-t5-small", "Local FLAN-T5 Small", "google/flan-t5-small", '["generation","judge","planner"]', '{"context_window":512,"max_output_tokens":160}'),
        ("model-local-hash-384", "Local Hash Embedding 384", "hash-embedding-384", '["embedding"]', '{"dimension":384,"batch_size":256}'),
        ("model-local-minilm-384", "Local MiniLM Embedding 384", "sentence-transformers/all-MiniLM-L6-v2", '["embedding"]', '{"dimension":384,"batch_size":64}'),
        ("model-local-lexical-reranker", "Local Lexical Reranker", "lexical-overlap", '["rerank"]', '{}'),
        ("model-local-distilbert", "Local DistilBERT Classifier", "query_classifier_distilbert", '["classifier"]', '{}'),
        ("model-local-t5-classifier", "Local T5-small Classifier", "query_classifier_t5", '["classifier"]', '{}'),
    ]
    bind = op.get_bind()
    statement = sa.text(
        """
        INSERT INTO model_deployments (
            id, name, provider, model, capabilities_json, limits_json, locality, enabled,
            health_status, metadata_json, created_at, updated_at
        ) VALUES (
            :deployment_id, :name, 'Local', :model, CAST(:capabilities AS JSONB), CAST(:limits AS JSONB),
            'local', TRUE, 'healthy', CAST(:metadata AS JSONB), NOW()::text, NOW()::text
        ) ON CONFLICT (id) DO NOTHING
        """
    )
    for deployment_id, name, model, capabilities, limits in rows:
        bind.execute(
            statement,
            {
                "deployment_id": deployment_id,
                "name": name,
                "model": model,
                "capabilities": capabilities,
                "limits": limits,
                "metadata": '{"builtin":true}',
            },
        )
