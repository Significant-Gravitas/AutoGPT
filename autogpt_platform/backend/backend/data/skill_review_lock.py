from prisma import Prisma
from prisma.models import SkillLearningSource

from backend.data.db import query_raw_with_schema
from backend.data.skill_learning import LearningSourceRecord


async def lock_review_source(
    client: Prisma, user_id: str, source_id: str
) -> LearningSourceRecord | None:
    rows = await query_raw_with_schema(
        'SELECT * FROM {schema_prefix}"SkillLearningSource" '
        'WHERE "id" = $1 AND "userId" = $2 FOR UPDATE',
        source_id,
        user_id,
        client=client,
        model=SkillLearningSource,
    )
    return LearningSourceRecord.from_db(rows[0]) if rows else None
