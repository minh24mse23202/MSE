"""Add encrypted Model Farm credential storage.

Revision ID: 20260713_03
Revises: 20260713_02
Create Date: 2026-07-13
"""
from alembic import op


revision = "20260713_03"
down_revision = "20260713_02"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE model_deployments
        ADD COLUMN IF NOT EXISTS credential_secrets_json JSONB NOT NULL DEFAULT '{}'::jsonb
        """
    )


def downgrade() -> None:
    op.drop_column("model_deployments", "credential_secrets_json")
