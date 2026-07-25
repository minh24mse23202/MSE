"""Add durable WixQA evaluation experiments.

Revision ID: 20260724_01
Revises: 20260721_01
Create Date: 2026-07-24
"""
from alembic import op


revision = "20260724_01"
down_revision = "20260721_01"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_runs (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            status TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            knowledge_base_name TEXT NOT NULL DEFAULT '',
            chat_configuration_id TEXT,
            retrieval_mode TEXT NOT NULL DEFAULT 'hybrid',
            top_k INTEGER NOT NULL DEFAULT 4,
            run_limit INTEGER NOT NULL DEFAULT 20,
            compare_baseline BOOLEAN NOT NULL DEFAULT FALSE,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            baseline_metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            route_distribution_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            baseline_route_distribution_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT,
            created_at TEXT NOT NULL,
            finished_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_cases (
            id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES evaluation_runs(id) ON DELETE CASCADE,
            record_id TEXT NOT NULL,
            question TEXT NOT NULL,
            expected_answer TEXT NOT NULL,
            complexity_label TEXT NOT NULL,
            adaptive_answer TEXT NOT NULL,
            static_answer TEXT NOT NULL DEFAULT '',
            adaptive_contexts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            static_contexts_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            adaptive_metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            static_metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TEXT NOT NULL
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluation_runs_created_at "
        "ON evaluation_runs(created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluation_cases_run_id "
        "ON evaluation_cases(run_id, created_at)"
    )
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS evaluation_experiments (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            knowledge_base_id TEXT NOT NULL,
            knowledge_base_name TEXT NOT NULL DEFAULT '',
            configuration_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            datasets_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            judge_deployment_id TEXT NOT NULL,
            quality_weights_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            max_cost_per_case DOUBLE PRECISION,
            max_average_latency_ms DOUBLE PRECISION,
            seed INTEGER NOT NULL DEFAULT 42,
            run_ids_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            leaderboard_json JSONB NOT NULL DEFAULT '[]'::jsonb,
            progress_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
            error TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            finished_at TEXT NOT NULL DEFAULT ''
        )
        """
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluation_experiments_created_at "
        "ON evaluation_experiments(created_at DESC)"
    )
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_evaluation_experiments_status "
        "ON evaluation_experiments(status, updated_at DESC)"
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS evaluation_experiments")
