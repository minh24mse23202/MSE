"""Reset custom Model Farm deployments for provider wizard rebuild.

Revision ID: 20260713_02
Revises: 20260713_01
Create Date: 2026-07-13
"""
from alembic import op
import sqlalchemy as sa


revision = "20260713_02"
down_revision = "20260713_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DELETE FROM model_usage_events")
    op.execute("DELETE FROM model_deployments WHERE COALESCE(metadata_json->>'builtin', 'false') <> 'true'")
    _seed_local_deployments()
    op.execute(
        """
        UPDATE chat_configurations
        SET metadata_json = jsonb_set(
            COALESCE(metadata_json, '{}'::jsonb),
            '{generator_deployment_id}',
            '"model-local-extractive"'::jsonb,
            true
        )
        WHERE COALESCE(metadata_json->>'generator_deployment_id', '') <> ''
          AND NOT EXISTS (
              SELECT 1 FROM model_deployments md
              WHERE md.id = metadata_json->>'generator_deployment_id'
          )
        """
    )
    op.execute(
        """
        UPDATE knowledge_bases
        SET metadata_json = jsonb_set(
            COALESCE(metadata_json, '{}'::jsonb),
            '{configuration,embedding_deployment_id}',
            '"model-local-hash-384"'::jsonb,
            true
        )
        WHERE COALESCE(metadata_json->'configuration'->>'embedding_deployment_id', '') <> ''
          AND NOT EXISTS (
              SELECT 1 FROM model_deployments md
              WHERE md.id = metadata_json->'configuration'->>'embedding_deployment_id'
          )
        """
    )
    op.execute(
        """
        UPDATE knowledge_index_versions
        SET embedding_deployment_id = 'model-local-hash-384'
        WHERE embedding_deployment_id <> ''
          AND NOT EXISTS (
              SELECT 1 FROM model_deployments md
              WHERE md.id = knowledge_index_versions.embedding_deployment_id
          )
        """
    )
    op.execute(
        """
        UPDATE chunk_embeddings
        SET embedding_deployment_id = 'model-local-hash-384'
        WHERE embedding_deployment_id <> ''
          AND NOT EXISTS (
              SELECT 1 FROM model_deployments md
              WHERE md.id = chunk_embeddings.embedding_deployment_id
          )
        """
    )


def downgrade() -> None:
    # This migration intentionally discards custom deployment records and usage.
    pass


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
    statement = sa.text(
        """
        INSERT INTO model_deployments (
            id, name, provider, model, capabilities_json, limits_json, locality, enabled,
            health_status, metadata_json, created_at, updated_at
        ) VALUES (
            :deployment_id, :name, 'Local', :model, CAST(:capabilities AS JSONB), CAST(:limits AS JSONB),
            'local', TRUE, 'healthy', '{"builtin": true}'::jsonb, NOW()::text, NOW()::text
        )
        ON CONFLICT (id) DO UPDATE SET
            name = EXCLUDED.name,
            provider = EXCLUDED.provider,
            model = EXCLUDED.model,
            capabilities_json = EXCLUDED.capabilities_json,
            limits_json = EXCLUDED.limits_json,
            locality = EXCLUDED.locality,
            enabled = TRUE,
            health_status = 'healthy',
            metadata_json = '{"builtin": true}'::jsonb,
            updated_at = NOW()::text
        """
    )
    bind = op.get_bind()
    for deployment_id, name, model, capabilities, limits in rows:
        bind.execute(
            statement,
            {
                "deployment_id": deployment_id,
                "name": name,
                "model": model,
                "capabilities": capabilities,
                "limits": limits,
            },
        )
