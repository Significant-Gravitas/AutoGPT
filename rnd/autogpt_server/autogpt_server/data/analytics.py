import logging

import prisma.enums
import prisma.types

logger = logging.getLogger(__name__)


async def log_raw_analytics(
    user_id: str,
    type: prisma.enums.AnalyticsType,
    data: dict,
    data_index: str,
):
    details = await prisma.models.AnalyticsDetails.prisma().create(
        data={
            "userId": user_id,
            "type": type,
            "data": prisma.Json(data),
            "dataIndex": data_index,
        }
    )
    return details


async def log_raw_metric(
    user_id: str,
    metric_name: prisma.enums.AnalyticsMetric,
    aggregation_type: prisma.enums.AggregationType,
    metric_value: float,
    data_string: str,
):
    if metric_value < 0:
        raise ValueError("metric_value must be non-negative")

    if aggregation_type == prisma.enums.AggregationType.NO_AGGREGATION:
        value_increment = metric_value
        counter_increment = 0
    elif aggregation_type in [
        prisma.enums.AggregationType.COUNT,
        prisma.enums.AggregationType.SUM,
    ]:
        value_increment = metric_value
        counter_increment = 1
    elif aggregation_type in [
        prisma.enums.AggregationType.AVG,
        prisma.enums.AggregationType.MAX,
        prisma.enums.AggregationType.MIN,
    ]:
        value_increment = 0  # These will be handled differently in a separate query
        counter_increment = 1
    else:
        raise ValueError(f"Unsupported aggregation_type: {aggregation_type}")

    result = await prisma.models.AnalyticsMetrics.prisma().upsert(
        data={
            "update": {
                "value": {"increment": value_increment},
                "aggregationCounter": {"increment": counter_increment},
            },
            "create": {
                "value": metric_value,
                "analyticMetric": metric_name,
                "userId": user_id,
                "dataString": data_string,
                "aggregationType": aggregation_type,
                "aggregationCounter": 1,
            },
        },
        where={
            "analyticMetric_userId_dataString_aggregationType": {
                "analyticMetric": metric_name,
                "userId": user_id,
                "dataString": data_string,
                "aggregationType": aggregation_type,
            }
        },
    )

    # For AVG, MAX, and MIN, we need to perform additional operations
    if aggregation_type in [
        prisma.enums.AggregationType.AVG,
        prisma.enums.AggregationType.MAX,
        prisma.enums.AggregationType.MIN,
    ]:
        existing = await prisma.models.AnalyticsMetrics.prisma().find_unique(
            where={
                "analyticMetric_userId_dataString_aggregationType": {
                    "analyticMetric": metric_name,
                    "userId": user_id,
                    "dataString": data_string,
                    "aggregationType": aggregation_type,
                }
            }
        )
        if existing:
            if aggregation_type == prisma.enums.AggregationType.AVG:
                new_value = (
                    existing.value * existing.aggregationCounter + metric_value
                ) / (existing.aggregationCounter + 1)
            elif aggregation_type == prisma.enums.AggregationType.MAX:
                new_value = max(existing.value, metric_value)
            else:  # MIN
                new_value = min(existing.value, metric_value)

            result = await prisma.models.AnalyticsMetrics.prisma().update(
                data={"value": new_value}, where={"id": existing.id}
            )
            if not result:
                raise ValueError(f"Failed to update metric: {existing.id}")
    return result
