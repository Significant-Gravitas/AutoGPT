"""Store search functionality with embeddings and pgvector using SQLAlchemy"""

import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence
from uuid import uuid4

from openai import AsyncOpenAI, OpenAIError
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Column, DateTime
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Index, String, UniqueConstraint, and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.declarative import declarative_base

from backend.util.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()

# Initialize async OpenAI client with proper configuration
_openai_client: Optional[AsyncOpenAI] = None


def get_openai_client() -> AsyncOpenAI:
    """Get or create the async OpenAI client with proper configuration."""
    global _openai_client
    if _openai_client is None:
        api_key = settings.secrets.openai_api_key
        if not api_key:
            logger.warning(
                "OpenAI API key not configured. Vector search will use fallback text search."
            )
            raise ValueError(
                "OpenAI API key is not configured. Please set OPENAI_API_KEY in environment."
            )
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


async def create_embedding(text: str) -> Optional[list[float]]:
    """Create an embedding for the given text using OpenAI's API.

    Args:
        text: The text to create an embedding for

    Returns:
        A list of floats representing the embedding, or None if creation fails
    """
    try:
        client = get_openai_client()
        response = await client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except ValueError as e:
        # API key not configured
        logger.error(f"OpenAI configuration error: {e}")
        return None
    except OpenAIError as e:
        # Handle specific OpenAI errors
        logger.error(f"OpenAI API error creating embedding: {e}")
        return None
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error creating embedding: {e}")
        return None


# SQLAlchemy models
Base = declarative_base()


class SubmissionStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"


class SearchFieldType(str, Enum):
    NAME = "NAME"
    DESCRIPTION = "DESCRIPTION"
    CATEGORIES = "CATEGORIES"
    SUBHEADING = "SUBHEADING"


class StoreAgentSearch(Base):
    __tablename__ = "StoreAgentSearch"

    id = Column(String, primary_key=True, default=lambda: str(uuid4()))
    createdAt = Column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updatedAt = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relations (foreign keys exist in DB but not modeled here)
    storeListingVersionId = Column(String, nullable=False)
    storeListingId = Column(String, nullable=False)

    # Searchable fields
    fieldName = Column(String, nullable=False)
    fieldValue = Column(String, nullable=False)

    # Vector embedding for similarity search
    # text-embedding-3-small produces 1536-dimensional embeddings
    embedding = Column(Vector(1536), nullable=False)

    # Metadata
    fieldType = Column(
        SQLEnum(SearchFieldType, name="SearchFieldType", schema="platform"),
        nullable=False,
    )
    submissionStatus = Column(
        SQLEnum(SubmissionStatus, name="SubmissionStatus", schema="platform"),
        nullable=False,
    )
    isAvailable = Column(Boolean, nullable=False)

    # Constraints and schema
    __table_args__ = (
        UniqueConstraint(
            "storeListingVersionId",
            "fieldName",
            name="_store_listing_version_field_unique",
        ),
        Index("ix_store_agent_search_listing_version", "storeListingVersionId"),
        Index("ix_store_agent_search_listing", "storeListingId"),
        Index("ix_store_agent_search_field_name", "fieldName"),
        Index("ix_store_agent_search_field_type", "fieldType"),
        Index(
            "ix_store_agent_search_status_available", "submissionStatus", "isAvailable"
        ),
        {"schema": "platform"},  # Specify the schema
    )


class StoreAgentSearchService:
    """Service class for Store Agent Search operations using SQLAlchemy"""

    def __init__(self, database_url: str):
        """Initialize the search service with async database connection"""
        # Parse the URL to handle schema and other params separately
        from urllib.parse import parse_qs, urlparse, urlunparse

        parsed = urlparse(database_url)
        query_params = parse_qs(parsed.query)

        # Extract schema if present
        schema = query_params.pop("schema", ["public"])[0]

        # Remove connect_timeout from query params (will be handled in connect_args)
        connect_timeout = query_params.pop("connect_timeout", [None])[0]

        # Rebuild query string without schema and connect_timeout
        new_query = "&".join([f"{k}={v[0]}" for k, v in query_params.items()])

        # Rebuild URL without schema and connect_timeout parameters
        clean_url = urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            )
        )

        # Convert to async URL for asyncpg
        if clean_url.startswith("postgresql://"):
            clean_url = clean_url.replace("postgresql://", "postgresql+asyncpg://")
        elif clean_url.startswith("postgres://"):
            clean_url = clean_url.replace("postgres://", "postgresql+asyncpg://")

        # Build connect_args
        connect_args: dict[str, Any] = {"server_settings": {"search_path": schema}}

        # Add timeout if present (asyncpg uses 'timeout' not 'connect_timeout')
        if connect_timeout:
            connect_args["timeout"] = float(connect_timeout)

        # Create engine with schema in connect_args
        self.engine = create_async_engine(
            clean_url,
            echo=False,  # Set to True for debugging SQL queries
            future=True,
            connect_args=connect_args,
        )

        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def create_search_record(
        self,
        store_listing_version_id: str,
        store_listing_id: str,
        field_name: str,
        field_value: str,
        embedding: list[float],
        field_type: SearchFieldType,
        submission_status: SubmissionStatus,
        is_available: bool,
    ) -> Optional[StoreAgentSearch]:
        """Create a new search record with embedding.

        Returns:
            The created search record or None if creation fails.
        """
        try:
            async with self.async_session() as session:
                search_record = StoreAgentSearch(
                    storeListingVersionId=store_listing_version_id,
                    storeListingId=store_listing_id,
                    fieldName=field_name,
                    fieldValue=field_value,
                    embedding=embedding,
                    fieldType=field_type,
                    submissionStatus=submission_status,
                    isAvailable=is_available,
                )
                session.add(search_record)
                await session.commit()
                await session.refresh(search_record)
                return search_record
        except Exception as e:
            logger.error(f"Failed to create search record: {e}")
            return None

    async def batch_create_search_records(
        self, records: list[dict]
    ) -> list[StoreAgentSearch]:
        """Batch create multiple search records"""
        async with self.async_session() as session:
            search_records = []
            for record in records:
                search_record = StoreAgentSearch(
                    storeListingVersionId=record["storeListingVersionId"],
                    storeListingId=record["storeListingId"],
                    fieldName=record["fieldName"],
                    fieldValue=record["fieldValue"],
                    embedding=record["embedding"],
                    fieldType=record.get("fieldType", SearchFieldType.NAME),
                    submissionStatus=record["submissionStatus"],
                    isAvailable=record["isAvailable"],
                )
                session.add(search_record)
                search_records.append(search_record)

            await session.commit()
            return search_records

    async def search_by_embedding(
        self, query_embedding: List[float], limit: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Search for store agents using vector similarity.
        Returns the best matching store listings based on embedding similarity.

        Args:
            query_embedding: The embedding vector to search with
            limit: Maximum number of results to return

        Returns:
            List of matching store agents with similarity scores

        Raises:
            Exception: If the database query fails
        """
        if not query_embedding:
            logger.warning("Empty embedding provided for search")
            return []

        try:
            async with self.async_session() as session:
                # Use parameterized query to prevent SQL injection
                query = text(
                    """
                WITH similarity_scores AS (
                    SELECT
                        sas."storeListingId",
                        MIN(sas.embedding <=> CAST(:embedding AS vector)) AS similarity_score
                    FROM platform."StoreAgentSearch" sas
                    WHERE
                        sas."submissionStatus" = 'APPROVED'
                        AND sas."isAvailable" = true
                    GROUP BY sas."storeListingId"
                    ORDER BY similarity_score
                    LIMIT :limit
                )
                SELECT
                    sa.listing_id as id,
                    sa.slug,
                    sa.agent_name,
                    sa.agent_image,
                    sa.description,
                    sa.sub_heading,
                    sa.featured,
                    sa.runs,
                    sa.rating,
                    sa.creator_username,
                    sa.creator_avatar,
                    ss.similarity_score
                FROM similarity_scores ss
                INNER JOIN platform."StoreAgent" sa
                    ON sa.listing_id = ss."storeListingId"
                ORDER BY ss.similarity_score;
            """
                )

                # Format embedding as PostgreSQL array safely
                embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

                result = await session.execute(
                    query, {"embedding": embedding_str, "limit": limit}
                )

                rows = result.fetchall()

                # Convert rows to dictionaries
                return [dict(row._mapping) for row in rows]
        except Exception as e:
            logger.error(f"Vector search query failed: {e}")
            # Return empty results instead of propagating error
            # This allows fallback to text search
            return []

    async def get_search_records(
        self,
        store_listing_version_id: Optional[str] = None,
        field_name: Optional[str] = None,
        is_available: Optional[bool] = None,
    ) -> Sequence[StoreAgentSearch]:
        """
        Get search records using SQLAlchemy ORM
        """
        async with self.async_session() as session:
            stmt = select(StoreAgentSearch)

            # Build filters
            filters = []
            if store_listing_version_id:
                filters.append(
                    StoreAgentSearch.storeListingVersionId == store_listing_version_id
                )
            if field_name:
                filters.append(StoreAgentSearch.fieldName == field_name)
            if is_available is not None:
                filters.append(StoreAgentSearch.isAvailable == is_available)

            if filters:
                stmt = stmt.where(and_(*filters))

            result = await session.execute(stmt)
            return result.scalars().all()

    async def update_search_embeddings(
        self, store_listing_version_id: str, updates: Dict[str, List[float]]
    ) -> None:
        """Update embeddings for existing search records"""
        async with self.async_session() as session:
            for field_name, embedding in updates.items():
                # For vector updates, we still need raw SQL due to pgvector
                # Use $ parameters for asyncpg
                query = text(
                    """
                    UPDATE platform."StoreAgentSearch"
                    SET embedding = CAST(:embedding AS vector),
                        "updatedAt" = CAST(:updated_at AS TIMESTAMPTZ)
                    WHERE "storeListingVersionId" = :version_id
                    AND "fieldName" = :field_name
                """
                )

                embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                await session.execute(
                    query,
                    {
                        "embedding": embedding_str,
                        "updated_at": datetime.now(timezone.utc),
                        "version_id": store_listing_version_id,
                        "field_name": field_name,
                    },
                )

            await session.commit()

    async def delete_search_records(self, store_listing_version_id: str) -> None:
        """Delete all search records for a store listing version using SQLAlchemy ORM"""
        async with self.async_session() as session:
            # Use SQLAlchemy ORM for deletion
            stmt = select(StoreAgentSearch).where(
                StoreAgentSearch.storeListingVersionId == store_listing_version_id
            )
            result = await session.execute(stmt)
            records = result.scalars().all()

            for record in records:
                await session.delete(record)

            await session.commit()

    async def upsert_search_record(
        self,
        store_listing_version_id: str,
        store_listing_id: str,
        field_name: str,
        field_value: str,
        embedding: List[float],
        field_type: SearchFieldType,
        submission_status: SubmissionStatus,
        is_available: bool,
    ) -> Optional[StoreAgentSearch]:
        """Upsert a search record (update if exists, create if not).

        Returns:
            The upserted search record or None if operation fails.
        """
        try:
            async with self.async_session() as session:
                # Check if record exists
                stmt = select(StoreAgentSearch).where(
                    and_(
                        StoreAgentSearch.storeListingVersionId
                        == store_listing_version_id,
                        StoreAgentSearch.fieldName == field_name,
                    )
                )
                result = await session.execute(stmt)
                existing_record = result.scalar_one_or_none()

                if existing_record:
                    # Update existing record
                    existing_record.fieldValue = field_value  # type: ignore[attr-defined]
                    existing_record.embedding = embedding  # type: ignore[attr-defined]
                    existing_record.fieldType = field_type  # type: ignore[attr-defined]
                    existing_record.submissionStatus = submission_status  # type: ignore[attr-defined]
                    existing_record.isAvailable = is_available  # type: ignore[attr-defined]
                    existing_record.updatedAt = datetime.now(timezone.utc)  # type: ignore[attr-defined]

                    await session.commit()
                    await session.refresh(existing_record)
                    return existing_record
                else:
                    # Create new record
                    return await self.create_search_record(
                        store_listing_version_id=store_listing_version_id,
                        store_listing_id=store_listing_id,
                        field_name=field_name,
                        field_value=field_value,
                        embedding=embedding,
                        field_type=field_type,
                        submission_status=submission_status,
                        is_available=is_available,
                    )
        except Exception as e:
            logger.error(f"Failed to upsert search record: {e}")
            return None

    async def close(self):
        """Close the database connection"""
        await self.engine.dispose()
