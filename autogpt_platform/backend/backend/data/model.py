from __future__ import annotations

import base64
import enum
import logging
from collections import defaultdict
from datetime import datetime, timezone
from json import JSONDecodeError
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Optional,
    TypeVar,
    cast,
    get_args,
)
from urllib.parse import urlparse
from uuid import uuid4

from prisma.enums import CreditTransactionType
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    SecretStr,
    field_serializer,
)
from pydantic_core import (
    CoreSchema,
    PydanticUndefined,
    PydanticUndefinedType,
    ValidationError,
    core_schema,
)
from typing_extensions import TypedDict

from backend.integrations.providers import ProviderName
from backend.util.json import loads as json_loads
from backend.util.settings import Secrets

# Type alias for any provider name (including custom ones)
AnyProviderName = str  # Will be validated as ProviderName at runtime


class User(BaseModel):
    """Application-layer User model with snake_case convention."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email address")
    email_verified: bool = Field(default=True, description="Whether email is verified")
    name: Optional[str] = Field(None, description="User display name")
    created_at: datetime = Field(..., description="When user was created")
    updated_at: datetime = Field(..., description="When user was last updated")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="User metadata as dict"
    )
    integrations: str = Field(default="", description="Encrypted integrations data")
    stripe_customer_id: Optional[str] = Field(None, description="Stripe customer ID")
    top_up_config: Optional["AutoTopUpConfig"] = Field(
        None, description="Top up configuration"
    )

    # Notification preferences
    max_emails_per_day: int = Field(default=3, description="Maximum emails per day")
    notify_on_agent_run: bool = Field(default=True, description="Notify on agent run")
    notify_on_zero_balance: bool = Field(
        default=True, description="Notify on zero balance"
    )
    notify_on_low_balance: bool = Field(
        default=True, description="Notify on low balance"
    )
    notify_on_block_execution_failed: bool = Field(
        default=True, description="Notify on block execution failure"
    )
    notify_on_continuous_agent_error: bool = Field(
        default=True, description="Notify on continuous agent error"
    )
    notify_on_daily_summary: bool = Field(
        default=True, description="Notify on daily summary"
    )
    notify_on_weekly_summary: bool = Field(
        default=True, description="Notify on weekly summary"
    )
    notify_on_monthly_summary: bool = Field(
        default=True, description="Notify on monthly summary"
    )

    # User timezone for scheduling and time display
    timezone: str = Field(
        default="not-set",
        description="User timezone (IANA timezone identifier or 'not-set')",
    )

    @classmethod
    def from_db(cls, prisma_user: "PrismaUser") -> "User":
        """Convert a database User object to application User model."""
        # Handle metadata field - convert from JSON string or dict to dict
        metadata = {}
        if prisma_user.metadata:
            if isinstance(prisma_user.metadata, str):
                try:
                    metadata = json_loads(prisma_user.metadata)
                except (JSONDecodeError, TypeError):
                    metadata = {}
            elif isinstance(prisma_user.metadata, dict):
                metadata = prisma_user.metadata

        # Handle topUpConfig field
        top_up_config = None
        if prisma_user.topUpConfig:
            if isinstance(prisma_user.topUpConfig, str):
                try:
                    config_dict = json_loads(prisma_user.topUpConfig)
                    top_up_config = AutoTopUpConfig.model_validate(config_dict)
                except (JSONDecodeError, TypeError, ValueError):
                    top_up_config = None
            elif isinstance(prisma_user.topUpConfig, dict):
                try:
                    top_up_config = AutoTopUpConfig.model_validate(
                        prisma_user.topUpConfig
                    )
                except ValueError:
                    top_up_config = None

        return cls(
            id=prisma_user.id,
            email=prisma_user.email,
            email_verified=prisma_user.emailVerified or True,
            name=prisma_user.name,
            created_at=prisma_user.createdAt,
            updated_at=prisma_user.updatedAt,
            metadata=metadata,
            integrations=prisma_user.integrations or "",
            stripe_customer_id=prisma_user.stripeCustomerId,
            top_up_config=top_up_config,
            max_emails_per_day=prisma_user.maxEmailsPerDay or 3,
            notify_on_agent_run=prisma_user.notifyOnAgentRun or True,
            notify_on_zero_balance=prisma_user.notifyOnZeroBalance or True,
            notify_on_low_balance=prisma_user.notifyOnLowBalance or True,
            notify_on_block_execution_failed=prisma_user.notifyOnBlockExecutionFailed
            or True,
            notify_on_continuous_agent_error=prisma_user.notifyOnContinuousAgentError
            or True,
            notify_on_daily_summary=prisma_user.notifyOnDailySummary or True,
            notify_on_weekly_summary=prisma_user.notifyOnWeeklySummary or True,
            notify_on_monthly_summary=prisma_user.notifyOnMonthlySummary or True,
            timezone=prisma_user.timezone or "not-set",
        )


if TYPE_CHECKING:
    from prisma.models import User as PrismaUser

    from backend.data.block import BlockSchema

T = TypeVar("T")
logger = logging.getLogger(__name__)


class BlockSecret:
    def __init__(self, key: Optional[str] = None, value: Optional[str] = None):
        if value is not None:
            trimmed_value = value.strip()
            if value != trimmed_value:
                logger.debug(BlockSecret.TRIMMING_VALUE_MSG)
            self._value = trimmed_value
            return

        self._value = self.__get_secret(key)
        if self._value is None:
            raise ValueError(f"Secret {key} not found.")
        trimmed_value = self._value.strip()
        if self._value != trimmed_value:
            logger.debug(BlockSecret.TRIMMING_VALUE_MSG)
        self._value = trimmed_value

    TRIMMING_VALUE_MSG: ClassVar[str] = "Provided secret value got trimmed."
    STR: ClassVar[str] = "<secret>"
    SECRETS: ClassVar[Secrets] = Secrets()

    def __repr__(self):
        return BlockSecret.STR

    def __str__(self):
        return BlockSecret.STR

    @staticmethod
    def __get_secret(key: str | None):
        if not key or not hasattr(BlockSecret.SECRETS, key):
            return None
        return getattr(BlockSecret.SECRETS, key)

    def get_secret_value(self):
        trimmed_value = str(self._value).strip()
        if self._value != trimmed_value:
            logger.info(BlockSecret.TRIMMING_VALUE_MSG)
        return trimmed_value

    @classmethod
    def parse_value(cls, value: Any) -> BlockSecret:
        if isinstance(value, BlockSecret):
            return value
        return BlockSecret(value=value)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> dict[str, Any]:
        return {
            "type": "string",
        }

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        validate_fun = core_schema.no_info_plain_validator_function(cls.parse_value)
        return core_schema.json_or_python_schema(
            json_schema=validate_fun,
            python_schema=validate_fun,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda val: BlockSecret.STR
            ),
        )


def SecretField(
    value: Optional[str] = None,
    key: Optional[str] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    placeholder: Optional[str] = None,
    **kwargs,
) -> BlockSecret:
    return SchemaField(
        BlockSecret(key=key, value=value),
        title=title,
        description=description,
        placeholder=placeholder,
        secret=True,
        **kwargs,
    )


def SchemaField(
    default: T | PydanticUndefinedType = PydanticUndefined,
    *args,
    default_factory: Optional[Callable[[], T]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    placeholder: Optional[str] = None,
    advanced: Optional[bool] = None,
    secret: bool = False,
    exclude: bool = False,
    hidden: Optional[bool] = None,
    depends_on: Optional[list[str]] = None,
    ge: Optional[float] = None,
    le: Optional[float] = None,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    discriminator: Optional[str] = None,
    format: Optional[str] = None,
    json_schema_extra: Optional[dict[str, Any]] = None,
) -> T:
    if default is PydanticUndefined and default_factory is None:
        advanced = False
    elif advanced is None:
        advanced = True

    json_schema_extra = {
        k: v
        for k, v in {
            "placeholder": placeholder,
            "secret": secret,
            "advanced": advanced,
            "hidden": hidden,
            "depends_on": depends_on,
            "format": format,
            **(json_schema_extra or {}),
        }.items()
        if v is not None
    }

    return Field(
        default,
        *args,
        default_factory=default_factory,
        title=title,
        description=description,
        exclude=exclude,
        ge=ge,
        le=le,
        min_length=min_length,
        max_length=max_length,
        discriminator=discriminator,
        json_schema_extra=json_schema_extra,
    )  # type: ignore


class _BaseCredentials(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    title: Optional[str] = None

    @field_serializer("*")
    def dump_secret_strings(value: Any, _info):
        if isinstance(value, SecretStr):
            return value.get_secret_value()
        return value


class OAuth2Credentials(_BaseCredentials):
    type: Literal["oauth2"] = "oauth2"
    username: Optional[str] = None
    """Username of the third-party service user that these credentials belong to"""
    access_token: SecretStr
    access_token_expires_at: Optional[int] = None
    """Unix timestamp (seconds) indicating when the access token expires (if at all)"""
    refresh_token: Optional[SecretStr] = None
    refresh_token_expires_at: Optional[int] = None
    """Unix timestamp (seconds) indicating when the refresh token expires (if at all)"""
    scopes: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def auth_header(self) -> str:
        return f"Bearer {self.access_token.get_secret_value()}"


class APIKeyCredentials(_BaseCredentials):
    type: Literal["api_key"] = "api_key"
    api_key: SecretStr
    expires_at: Optional[int] = Field(
        default=None,
        description="Unix timestamp (seconds) indicating when the API key expires (if at all)",
    )
    """Unix timestamp (seconds) indicating when the API key expires (if at all)"""

    def auth_header(self) -> str:
        return f"Bearer {self.api_key.get_secret_value()}"


class UserPasswordCredentials(_BaseCredentials):
    type: Literal["user_password"] = "user_password"
    username: SecretStr
    password: SecretStr

    def auth_header(self) -> str:
        # Converting the string to bytes using encode()
        # Base64 encoding it with base64.b64encode()
        # Converting the resulting bytes back to a string with decode()
        return f"Basic {base64.b64encode(f'{self.username.get_secret_value()}:{self.password.get_secret_value()}'.encode()).decode()}"


class HostScopedCredentials(_BaseCredentials):
    type: Literal["host_scoped"] = "host_scoped"
    host: str = Field(description="The host/URI pattern to match against request URLs")
    headers: dict[str, SecretStr] = Field(
        description="Key-value header map to add to matching requests",
        default_factory=dict,
    )

    def _extract_headers(self, headers: dict[str, SecretStr]) -> dict[str, str]:
        """Helper to extract secret values from headers."""
        return {key: value.get_secret_value() for key, value in headers.items()}

    @field_serializer("headers")
    def serialize_headers(self, headers: dict[str, SecretStr]) -> dict[str, str]:
        """Serialize headers by extracting secret values."""
        return self._extract_headers(headers)

    def get_headers_dict(self) -> dict[str, str]:
        """Get headers with secret values extracted."""
        return self._extract_headers(self.headers)

    def auth_header(self) -> str:
        """Get authorization header for backward compatibility."""
        auth_headers = self.get_headers_dict()
        if "Authorization" in auth_headers:
            return auth_headers["Authorization"]
        return ""

    def matches_url(self, url: str) -> bool:
        """Check if this credential should be applied to the given URL."""

        parsed_url = urlparse(url)
        # Extract hostname without port
        request_host = parsed_url.hostname
        if not request_host:
            return False

        # Simple host matching - exact match or wildcard subdomain match
        if self.host == request_host:
            return True

        # Support wildcard matching (e.g., "*.example.com" matches "api.example.com")
        if self.host.startswith("*."):
            domain = self.host[2:]  # Remove "*."
            return request_host.endswith(f".{domain}") or request_host == domain

        return False


Credentials = Annotated[
    OAuth2Credentials
    | APIKeyCredentials
    | UserPasswordCredentials
    | HostScopedCredentials,
    Field(discriminator="type"),
]


CredentialsType = Literal["api_key", "oauth2", "user_password", "host_scoped"]


class OAuthState(BaseModel):
    token: str
    provider: str
    expires_at: int
    code_verifier: Optional[str] = None
    """Unix timestamp (seconds) indicating when this OAuth state expires"""
    scopes: list[str]


class UserMetadata(BaseModel):
    integration_credentials: list[Credentials] = Field(default_factory=list)
    """⚠️ Deprecated; use `UserIntegrations.credentials` instead"""
    integration_oauth_states: list[OAuthState] = Field(default_factory=list)
    """⚠️ Deprecated; use `UserIntegrations.oauth_states` instead"""


class UserMetadataRaw(TypedDict, total=False):
    integration_credentials: list[dict]
    """⚠️ Deprecated; use `UserIntegrations.credentials` instead"""
    integration_oauth_states: list[dict]
    """⚠️ Deprecated; use `UserIntegrations.oauth_states` instead"""


class UserIntegrations(BaseModel):

    class ManagedCredentials(BaseModel):
        """Integration credentials managed by us, rather than by the user"""

        ayrshare_profile_key: Optional[SecretStr] = None

        @field_serializer("*")
        def dump_secret_strings(value: Any, _info):
            if isinstance(value, SecretStr):
                return value.get_secret_value()
            return value

    managed_credentials: ManagedCredentials = Field(default_factory=ManagedCredentials)
    credentials: list[Credentials] = Field(default_factory=list)
    oauth_states: list[OAuthState] = Field(default_factory=list)


CP = TypeVar("CP", bound=ProviderName)
CT = TypeVar("CT", bound=CredentialsType)


def is_credentials_field_name(field_name: str) -> bool:
    return field_name == "credentials" or field_name.endswith("_credentials")


class CredentialsMetaInput(BaseModel, Generic[CP, CT]):
    id: str
    title: Optional[str] = None
    provider: CP
    type: CT

    @classmethod
    def allowed_providers(cls) -> tuple[ProviderName, ...] | None:
        return get_args(cls.model_fields["provider"].annotation)

    @classmethod
    def allowed_cred_types(cls) -> tuple[CredentialsType, ...]:
        return get_args(cls.model_fields["type"].annotation)

    @classmethod
    def validate_credentials_field_schema(cls, model: type["BlockSchema"]):
        """Validates the schema of a credentials input field"""
        field_name = next(
            name for name, type in model.get_credentials_fields().items() if type is cls
        )
        field_schema = model.jsonschema()["properties"][field_name]
        try:
            schema_extra = CredentialsFieldInfo[CP, CT].model_validate(field_schema)
        except ValidationError as e:
            if "Field required [type=missing" not in str(e):
                raise

            raise TypeError(
                "Field 'credentials' JSON schema lacks required extra items: "
                f"{field_schema}"
            ) from e

        providers = cls.allowed_providers()
        if (
            providers is not None
            and len(providers) > 1
            and not schema_extra.discriminator
        ):
            raise TypeError(
                f"Multi-provider CredentialsField '{field_name}' "
                "requires discriminator!"
            )

    @staticmethod
    def _add_json_schema_extra(schema: dict, model_class: type):
        # Use model_class for allowed_providers/cred_types
        if hasattr(model_class, "allowed_providers") and hasattr(
            model_class, "allowed_cred_types"
        ):
            allowed_providers = model_class.allowed_providers()
            # If no specific providers (None), allow any string
            if allowed_providers is None:
                schema["credentials_provider"] = ["string"]  # Allow any string provider
            else:
                schema["credentials_provider"] = allowed_providers
            schema["credentials_types"] = model_class.allowed_cred_types()
        # Do not return anything, just mutate schema in place

    model_config = ConfigDict(
        json_schema_extra=_add_json_schema_extra,  # type: ignore
    )


def _extract_host_from_url(url: str) -> str:
    """Extract host from URL for grouping host-scoped credentials."""
    try:
        parsed = urlparse(url)
        return parsed.hostname or url
    except Exception:
        return ""


class CredentialsFieldInfo(BaseModel, Generic[CP, CT]):
    # TODO: move discrimination mechanism out of CredentialsField (frontend + backend)
    provider: frozenset[CP] = Field(..., alias="credentials_provider")
    supported_types: frozenset[CT] = Field(..., alias="credentials_types")
    required_scopes: Optional[frozenset[str]] = Field(None, alias="credentials_scopes")
    discriminator: Optional[str] = None
    discriminator_mapping: Optional[dict[str, CP]] = None
    discriminator_values: set[Any] = Field(default_factory=set)

    @classmethod
    def combine(
        cls, *fields: tuple[CredentialsFieldInfo[CP, CT], T]
    ) -> dict[str, tuple[CredentialsFieldInfo[CP, CT], set[T]]]:
        """
        Combines multiple CredentialsFieldInfo objects into as few as possible.

        Rules:
        - Items can only be combined if they have the same supported credentials types
          and the same supported providers.
        - When combining items, the `required_scopes` of the result is a join
          of the `required_scopes` of the original items.

        Params:
            *fields: (CredentialsFieldInfo, key) objects to group and combine

        Returns:
            A sequence of tuples containing combined CredentialsFieldInfo objects and
            the set of keys of the respective original items that were grouped together.
        """
        if not fields:
            return {}

        # Group fields by their provider and supported_types
        # For HTTP host-scoped credentials, also group by host
        grouped_fields: defaultdict[
            tuple[frozenset[CP], frozenset[CT]],
            list[tuple[T, CredentialsFieldInfo[CP, CT]]],
        ] = defaultdict(list)

        for field, key in fields:
            if field.provider == frozenset([ProviderName.HTTP]):
                # HTTP host-scoped credentials can have different hosts that reqires different credential sets.
                # Group by host extracted from the URL
                providers = frozenset(
                    [cast(CP, "http")]
                    + [
                        cast(CP, _extract_host_from_url(str(value)))
                        for value in field.discriminator_values
                    ]
                )
            else:
                providers = frozenset(field.provider)

            group_key = (providers, frozenset(field.supported_types))
            grouped_fields[group_key].append((key, field))

        # Combine fields within each group
        result: dict[str, tuple[CredentialsFieldInfo[CP, CT], set[T]]] = {}

        for key, group in grouped_fields.items():
            # Start with the first field in the group
            _, combined = group[0]

            # Track the keys that were combined
            combined_keys = {key for key, _ in group}

            # Combine required_scopes from all fields in the group
            all_scopes = set()
            for _, field in group:
                if field.required_scopes:
                    all_scopes.update(field.required_scopes)

            # Combine discriminator_values from all fields in the group (removing duplicates)
            all_discriminator_values = []
            for _, field in group:
                for value in field.discriminator_values:
                    if value not in all_discriminator_values:
                        all_discriminator_values.append(value)

            # Generate the key for the combined result
            providers_key, supported_types_key = key
            group_key = (
                "-".join(sorted(providers_key))
                + "_"
                + "-".join(sorted(supported_types_key))
                + "_credentials"
            )

            result[group_key] = (
                CredentialsFieldInfo[CP, CT](
                    credentials_provider=combined.provider,
                    credentials_types=combined.supported_types,
                    credentials_scopes=frozenset(all_scopes) or None,
                    discriminator=combined.discriminator,
                    discriminator_mapping=combined.discriminator_mapping,
                    discriminator_values=set(all_discriminator_values),
                ),
                combined_keys,
            )

        return result

    def discriminate(self, discriminator_value: Any) -> CredentialsFieldInfo:
        if not (self.discriminator and self.discriminator_mapping):
            return self

        return CredentialsFieldInfo(
            credentials_provider=frozenset(
                [self.discriminator_mapping[discriminator_value]]
            ),
            credentials_types=self.supported_types,
            credentials_scopes=self.required_scopes,
            discriminator=self.discriminator,
            discriminator_mapping=self.discriminator_mapping,
            discriminator_values=self.discriminator_values,
        )


def CredentialsField(
    required_scopes: set[str] = set(),
    *,
    discriminator: Optional[str] = None,
    discriminator_mapping: Optional[dict[str, Any]] = None,
    discriminator_values: Optional[set[Any]] = None,
    title: Optional[str] = None,
    description: Optional[str] = None,
    **kwargs,
) -> CredentialsMetaInput:
    """
    `CredentialsField` must and can only be used on fields named `credentials`.
    This is enforced by the `BlockSchema` base class.
    """

    field_schema_extra = {
        k: v
        for k, v in {
            "credentials_scopes": list(required_scopes) or None,
            "discriminator": discriminator,
            "discriminator_mapping": discriminator_mapping,
            "discriminator_values": discriminator_values,
        }.items()
        if v is not None
    }

    # Merge any json_schema_extra passed in kwargs
    if "json_schema_extra" in kwargs:
        extra_schema = kwargs.pop("json_schema_extra")
        field_schema_extra.update(extra_schema)

    return Field(
        title=title,
        description=description,
        json_schema_extra=field_schema_extra,  # validated on BlockSchema init
        **kwargs,
    )


class ContributorDetails(BaseModel):
    name: str = Field(title="Name", description="The name of the contributor.")


class TopUpType(enum.Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"
    UNCATEGORIZED = "UNCATEGORIZED"


class AutoTopUpConfig(BaseModel):
    amount: int
    """Amount of credits to top up."""
    threshold: int
    """Threshold to trigger auto top up."""


class UserTransaction(BaseModel):
    transaction_key: str = ""
    transaction_time: datetime = datetime.min.replace(tzinfo=timezone.utc)
    transaction_type: CreditTransactionType = CreditTransactionType.USAGE
    amount: int = 0
    running_balance: int = 0
    current_balance: int = 0
    description: str | None = None
    usage_graph_id: str | None = None
    usage_execution_id: str | None = None
    usage_node_count: int = 0
    usage_start_time: datetime = datetime.max.replace(tzinfo=timezone.utc)
    user_id: str
    user_email: str | None = None
    reason: str | None = None
    admin_email: str | None = None
    extra_data: str | None = None


class TransactionHistory(BaseModel):
    transactions: list[UserTransaction]
    next_transaction_time: datetime | None


class RefundRequest(BaseModel):
    id: str
    user_id: str
    transaction_key: str
    amount: int
    reason: str
    result: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime


class NodeExecutionStats(BaseModel):
    """Execution statistics for a node execution."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    error: Optional[BaseException | str] = None
    walltime: float = 0
    cputime: float = 0
    input_size: int = 0
    output_size: int = 0
    llm_call_count: int = 0
    llm_retry_count: int = 0
    input_token_count: int = 0
    output_token_count: int = 0
    extra_cost: int = 0
    extra_steps: int = 0
    # Moderation fields
    cleared_inputs: Optional[dict[str, list[str]]] = None
    cleared_outputs: Optional[dict[str, list[str]]] = None

    def __iadd__(self, other: "NodeExecutionStats") -> "NodeExecutionStats":
        """Mutate this instance by adding another NodeExecutionStats."""
        if not isinstance(other, NodeExecutionStats):
            return NotImplemented

        stats_dict = other.model_dump()
        current_stats = self.model_dump()

        for key, value in stats_dict.items():
            if key not in current_stats:
                # Field doesn't exist yet, just set it
                setattr(self, key, value)
            elif isinstance(value, dict) and isinstance(current_stats[key], dict):
                current_stats[key].update(value)
                setattr(self, key, current_stats[key])
            elif isinstance(value, (int, float)) and isinstance(
                current_stats[key], (int, float)
            ):
                setattr(self, key, current_stats[key] + value)
            elif isinstance(value, list) and isinstance(current_stats[key], list):
                current_stats[key].extend(value)
                setattr(self, key, current_stats[key])
            else:
                setattr(self, key, value)

        return self


class GraphExecutionStats(BaseModel):
    """Execution statistics for a graph execution."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    error: Optional[Exception | str] = None
    walltime: float = Field(
        default=0, description="Time between start and end of run (seconds)"
    )
    cputime: float = 0
    nodes_walltime: float = Field(
        default=0, description="Total node execution time (seconds)"
    )
    nodes_cputime: float = 0
    node_count: int = Field(default=0, description="Total number of node executions")
    node_error_count: int = Field(
        default=0, description="Total number of errors generated"
    )
    cost: int = Field(default=0, description="Total execution cost (cents)")
    activity_status: Optional[str] = Field(
        default=None, description="AI-generated summary of what the agent did"
    )


class UserExecutionSummaryStats(BaseModel):
    """Summary of user statistics for a specific user."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
    )

    total_credits_used: float = Field(default=0)
    total_executions: int = Field(default=0)
    successful_runs: int = Field(default=0)
    failed_runs: int = Field(default=0)
    most_used_agent: str = Field(default="")
    total_execution_time: float = Field(default=0)
    average_execution_time: float = Field(default=0)
    cost_breakdown: dict[str, float] = Field(default_factory=dict)
