from pydantic import BaseModel, Field


class LatencyStats(BaseModel):
    series: list[int] | list[float]

    @property
    def min(self) -> int | float:
        return min(self.series)

    @property
    def max(self) -> int | float:
        return max(self.series)

    @property
    def avg(self) -> float:
        return sum(self.series) / len(self.series)

    @property
    def median(self) -> int | float:
        sorted_series = sorted(self.series)
        n = len(sorted_series)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_series[mid - 1] + sorted_series[mid]) / 2
        else:
            return sorted_series[mid]


class PullRequestMetrics(BaseModel):
    days_since_opened: int
    days_since_author_comment: int
    days_since_maintainer_comment: int | None
    days_since_maintainer_review: int | None
    days_until_first_maintainer_comment: int | None
    days_until_first_maintainer_review: int | None

    author_response_stats: LatencyStats | None = Field(
        ...,
        description="Stats on every time the author responded to a maintainer's comment"
    )
    maintainer_response_stats: LatencyStats | None = Field(
        ...,
        description="Stats on every time a maintainer responded to an author's comment"
    )
