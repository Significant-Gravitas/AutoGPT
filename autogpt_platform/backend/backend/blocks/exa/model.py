from typing import Any, Dict, List, Literal, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

# Enum definitions based on available options
class WebsetStatus(str, Enum):
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"

class WebsetSearchStatus(str, Enum):
    CREATED = "created"
    # Add more if known, based on example it's "created"

class ImportStatus(str, Enum):
    PENDING = "pending"
    # Add more if known

class ImportFormat(str, Enum):
    CSV = "csv"
    # Add more if known

class EnrichmentStatus(str, Enum):
    PENDING = "pending"
    # Add more if known

class EnrichmentFormat(str, Enum):
    TEXT = "text"
    # Add more if known

class MonitorStatus(str, Enum):
    ENABLED = "enabled"
    # Add more if known

class MonitorBehaviorType(str, Enum):
    SEARCH = "search"
    # Add more if known

class MonitorRunStatus(str, Enum):
    CREATED = "created"
    # Add more if known

class CanceledReason(str, Enum):
    WEBSET_DELETED = "webset_deleted"
    # Add more if known

class FailedReason(str, Enum):
    INVALID_FORMAT = "invalid_format"
    # Add more if known

class Confidence(str, Enum):
    HIGH = "high"
    # Add more if known

# Nested models

class Entity(BaseModel):
    type: str

class Criterion(BaseModel):
    description: str
    successRate: Optional[int] = None

class ExcludeItem(BaseModel):
    source: str = Field(..., example="import")
    id: str

class Relationship(BaseModel):
    definition: str
    limit: Optional[float] = None

class ScopeItem(BaseModel):
    source: str = Field(..., example="import")
    id: str
    relationship: Optional[Relationship] = None

class Progress(BaseModel):
    found: int
    analyzed: int
    completion: int
    timeLeft: int

class Bounds(BaseModel):
    min: int
    max: int

class Expected(BaseModel):
    total: int
    confidence: str = Field(..., example="high")  # Use str or Confidence enum
    bounds: Bounds

class Recall(BaseModel):
    expected: Expected
    reasoning: str

class WebsetSearch(BaseModel):
    id: str
    object: str = Field(default="webset_search", const=True)
    status: str = Field(..., example="created")  # Or use WebsetSearchStatus
    websetId: str
    query: str
    entity: Entity
    criteria: List[Criterion]
    count: int
    behavior: str = Field(..., example="override")
    exclude: List[ExcludeItem]
    scope: List[ScopeItem]
    progress: Progress
    recall: Recall
    metadata: Dict[str, Any] = Field(default_factory=dict)
    canceledAt: Optional[datetime] = None
    canceledReason: Optional[str] = Field(None, example="webset_deleted")  # Or use CanceledReason
    createdAt: datetime
    updatedAt: datetime

class ImportEntity(BaseModel):
    type: str

class Import(BaseModel):
    id: str
    object: str = Field(default="import", const=True)
    status: str = Field(..., example="pending")  # Or use ImportStatus
    format: str = Field(..., example="csv")  # Or use ImportFormat
    entity: ImportEntity
    title: str
    count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    failedReason: Optional[str] = Field(None, example="invalid_format")  # Or use FailedReason
    failedAt: Optional[datetime] = None
    failedMessage: Optional[str] = None
    createdAt: datetime
    updatedAt: datetime

class Option(BaseModel):
    label: str

class WebsetEnrichment(BaseModel):
    id: str
    object: str = Field(default="webset_enrichment", const=True)
    status: str = Field(..., example="pending")  # Or use EnrichmentStatus
    websetId: str
    title: str
    description: str
    format: str = Field(..., example="text")  # Or use EnrichmentFormat
    options: List[Option]
    instructions: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime
    updatedAt: datetime

class Cadence(BaseModel):
    cron: str
    timezone: str = Field(default="Etc/UTC")

class BehaviorConfig(BaseModel):
    query: Optional[str] = None
    criteria: Optional[List[Criterion]] = None
    entity: Optional[Entity] = None
    count: Optional[int] = None
    behavior: Optional[str] = Field(None, example="append")

class Behavior(BaseModel):
    type: str = Field(..., example="search")  # Or use MonitorBehaviorType
    config: BehaviorConfig

class MonitorRun(BaseModel):
    id: str
    object: str = Field(default="monitor_run", const=True)
    status: str = Field(..., example="created")  # Or use MonitorRunStatus
    monitorId: str
    type: str = Field(..., example="search")
    completedAt: Optional[datetime] = None
    failedAt: Optional[datetime] = None
    failedReason: Optional[str] = None
    canceledAt: Optional[datetime] = None
    createdAt: datetime
    updatedAt: datetime

class Monitor(BaseModel):
    id: str
    object: str = Field(default="monitor", const=True)
    status: str = Field(..., example="enabled")  # Or use MonitorStatus
    websetId: str
    cadence: Cadence
    behavior: Behavior
    lastRun: Optional[MonitorRun] = None
    nextRunAt: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    createdAt: datetime
    updatedAt: datetime

class Webset(BaseModel):
    id: str
    object: str = Field(default="webset", const=True)
    status: WebsetStatus
    externalId: Optional[str] = None
    title: Optional[str] = None
    searches: List[WebsetSearch]
    imports: List[Import]
    enrichments: List[WebsetEnrichment]
    monitors: List[Monitor]
    streams: List[Any]
    createdAt: datetime
    updatedAt: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ListWebsets(BaseModel):
    data: List[Webset]
    hasMore: bool
    nextCursor: Optional[str] = None