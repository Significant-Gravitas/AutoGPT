import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import prisma.types
from pydantic import BaseModel

from backend.data.db import query_raw_with_schema
from backend.util.json import SafeJson

logger = logging.getLogger(__name__)


class AccuracyAlertData(BaseModel):
    """Alert data when accuracy drops significantly."""

    graph_id: str
    user_id: Optional[str]
    drop_percent: float
    three_day_avg: float
    seven_day_avg: float
    detected_at: datetime


class AccuracyLatestData(BaseModel):
    """Latest execution accuracy data point."""

    date: datetime
    daily_score: Optional[float]
    three_day_avg: Optional[float]
    seven_day_avg: Optional[float]
    fourteen_day_avg: Optional[float]


class AccuracyTrendsResponse(BaseModel):
    """Response model for accuracy trends and alerts."""

    latest_data: AccuracyLatestData
    alert: Optional[AccuracyAlertData]
    historical_data: Optional[list[AccuracyLatestData]] = None


async def log_raw_analytics(
    user_id: str,
    type: str,
    data: dict,
    data_index: str,
):
    details = await prisma.models.AnalyticsDetails.prisma().create(
        data=prisma.types.AnalyticsDetailsCreateInput(
            userId=user_id,
            type=type,
            data=SafeJson(data),
            dataIndex=data_index,
        )
    )
    return details


async def log_raw_metric(
    user_id: str,
    metric_name: str,
    metric_value: float,
    data_string: str,
):
    if metric_value < 0:
        raise ValueError("metric_value must be non-negative")

    result = await prisma.models.AnalyticsMetrics.prisma().create(
        data=prisma.types.AnalyticsMetricsCreateInput(
            value=metric_value,
            analyticMetric=metric_name,
            userId=user_id,
            dataString=data_string,
        )
    )

    return result


async def get_accuracy_trends_and_alerts(
    graph_id: str,
    days_back: int = 30,
    user_id: Optional[str] = None,
    drop_threshold: float = 10.0,
    include_historical: bool = False,
) -> AccuracyTrendsResponse:
    """Get accuracy trends and detect alerts for a specific graph."""
    query_template = """
    WITH daily_scores AS (
        SELECT 
            DATE(e."createdAt") as execution_date,
            AVG(CASE 
                WHEN e.stats IS NOT NULL 
                AND e.stats::json->>'correctness_score' IS NOT NULL
                AND e.stats::json->>'correctness_score' != 'null'
                THEN (e.stats::json->>'correctness_score')::float * 100
                ELSE NULL 
            END) as daily_score
        FROM {schema_prefix}"AgentGraphExecution" e
        WHERE e."agentGraphId" = $1::text
            AND e."isDeleted" = false
            AND e."createdAt" >= $2::timestamp
            AND e."executionStatus" IN ('COMPLETED', 'FAILED', 'TERMINATED')
            {user_filter}
        GROUP BY DATE(e."createdAt")
        HAVING COUNT(*) >= 3  -- Need at least 3 executions per day
    ),
    trends AS (
        SELECT 
            execution_date,
            daily_score,
            AVG(daily_score) OVER (
                ORDER BY execution_date 
                ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
            ) as three_day_avg,
            AVG(daily_score) OVER (
                ORDER BY execution_date 
                ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
            ) as seven_day_avg,
            AVG(daily_score) OVER (
                ORDER BY execution_date 
                ROWS BETWEEN 13 PRECEDING AND CURRENT ROW
            ) as fourteen_day_avg
        FROM daily_scores
    )
    SELECT *,
        CASE 
            WHEN three_day_avg IS NOT NULL AND seven_day_avg IS NOT NULL AND seven_day_avg > 0
            THEN ((seven_day_avg - three_day_avg) / seven_day_avg * 100)
            ELSE NULL
        END as drop_percent
    FROM trends
    ORDER BY execution_date DESC
    {limit_clause}
    """

    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    params = [graph_id, start_date]
    user_filter = ""
    if user_id:
        user_filter = 'AND e."userId" = $3::text'
        params.append(user_id)

    # Determine limit clause
    limit_clause = "" if include_historical else "LIMIT 1"

    final_query = query_template.format(
        schema_prefix="{schema_prefix}",
        user_filter=user_filter,
        limit_clause=limit_clause,
    )

    result = await query_raw_with_schema(final_query, *params)

    if not result:
        return AccuracyTrendsResponse(
            latest_data=AccuracyLatestData(
                date=datetime.now(timezone.utc),
                daily_score=None,
                three_day_avg=None,
                seven_day_avg=None,
                fourteen_day_avg=None,
            ),
            alert=None,
        )

    latest = result[0]

    alert = None
    if (
        latest["drop_percent"] is not None
        and latest["drop_percent"] >= drop_threshold
        and latest["three_day_avg"] is not None
        and latest["seven_day_avg"] is not None
    ):
        alert = AccuracyAlertData(
            graph_id=graph_id,
            user_id=user_id,
            drop_percent=float(latest["drop_percent"]),
            three_day_avg=float(latest["three_day_avg"]),
            seven_day_avg=float(latest["seven_day_avg"]),
            detected_at=datetime.now(timezone.utc),
        )

    # Prepare historical data if requested
    historical_data = None
    if include_historical:
        historical_data = []
        for row in result:
            historical_data.append(
                AccuracyLatestData(
                    date=row["execution_date"],
                    daily_score=(
                        float(row["daily_score"])
                        if row["daily_score"] is not None
                        else None
                    ),
                    three_day_avg=(
                        float(row["three_day_avg"])
                        if row["three_day_avg"] is not None
                        else None
                    ),
                    seven_day_avg=(
                        float(row["seven_day_avg"])
                        if row["seven_day_avg"] is not None
                        else None
                    ),
                    fourteen_day_avg=(
                        float(row["fourteen_day_avg"])
                        if row["fourteen_day_avg"] is not None
                        else None
                    ),
                )
            )

    return AccuracyTrendsResponse(
        latest_data=AccuracyLatestData(
            date=latest["execution_date"],
            daily_score=(
                float(latest["daily_score"])
                if latest["daily_score"] is not None
                else None
            ),
            three_day_avg=(
                float(latest["three_day_avg"])
                if latest["three_day_avg"] is not None
                else None
            ),
            seven_day_avg=(
                float(latest["seven_day_avg"])
                if latest["seven_day_avg"] is not None
                else None
            ),
            fourteen_day_avg=(
                float(latest["fourteen_day_avg"])
                if latest["fourteen_day_avg"] is not None
                else None
            ),
        ),
        alert=alert,
        historical_data=historical_data,
    )


class MarketplaceGraphData(BaseModel):
    """Data structure for marketplace graph monitoring."""

    graph_id: str
    user_id: Optional[str]
    execution_count: int


async def get_marketplace_graphs_for_monitoring(
    days_back: int = 30,
    min_executions: int = 10,
) -> list[MarketplaceGraphData]:
    """Get published marketplace graphs with recent executions for monitoring."""
    query_template = """
    WITH marketplace_graphs AS (
        SELECT DISTINCT 
            slv."agentGraphId" as graph_id,
            slv."agentGraphVersion" as graph_version
        FROM {schema_prefix}"StoreListing" sl
        JOIN {schema_prefix}"StoreListingVersion" slv ON sl."activeVersionId" = slv."id"
        WHERE sl."hasApprovedVersion" = true
            AND sl."isDeleted" = false
    )
    SELECT DISTINCT 
        mg.graph_id,
        NULL as user_id,  -- Marketplace graphs don't have a specific user_id for monitoring
        COUNT(*) as execution_count
    FROM marketplace_graphs mg
    JOIN {schema_prefix}"AgentGraphExecution" e ON e."agentGraphId" = mg.graph_id
    WHERE e."createdAt" >= $1::timestamp
        AND e."isDeleted" = false
        AND e."executionStatus" IN ('COMPLETED', 'FAILED', 'TERMINATED')
    GROUP BY mg.graph_id
    HAVING COUNT(*) >= $2
    ORDER BY execution_count DESC
    """
    start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    result = await query_raw_with_schema(query_template, start_date, min_executions)

    return [
        MarketplaceGraphData(
            graph_id=row["graph_id"],
            user_id=row["user_id"],
            execution_count=int(row["execution_count"]),
        )
        for row in result
    ]
