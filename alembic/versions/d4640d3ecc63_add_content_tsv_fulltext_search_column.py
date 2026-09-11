"""add content_tsv fulltext search column

Revision ID: d4640d3ecc63
Revises: 9b17ea7bb659
Create Date: 2026-09-11 14:09:08.674223

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4640d3ecc63'
down_revision: Union[str, None] = '9b17ea7bb659'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Generated STORED column: computed automatically from `content` on
    # insert/update by Postgres itself, so existing rows get a tsvector
    # immediately on migration with no manual backfill step. PG16 supports
    # generated tsvector columns natively (to_tsvector with an explicit
    # regconfig literal like 'english' is IMMUTABLE).
    op.execute(
        "ALTER TABLE document_chunks "
        "ADD COLUMN content_tsv tsvector "
        "GENERATED ALWAYS AS (to_tsvector('english', content)) STORED"
    )
    op.execute(
        "CREATE INDEX document_chunks_content_tsv_idx "
        "ON document_chunks USING gin (content_tsv)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS document_chunks_content_tsv_idx")
    op.execute("ALTER TABLE document_chunks DROP COLUMN IF EXISTS content_tsv")
