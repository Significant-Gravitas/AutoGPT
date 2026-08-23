import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from strenum import StrEnum

BASE_CHAIN_ID = 8453
BASE_USDC_ADDRESS = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"
USDC_QUANTUM = Decimal("0.000001")


class TaskMarketMode(StrEnum):
    BOUNTY = "bounty"
    CLAIM = "claim"
    PITCH = "pitch"
    BENCHMARK = "benchmark"


class TaskMarketNetwork(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = "Base"
    chain_id: int = BASE_CHAIN_ID
    currency: str = "USDC"
    usdc_contract: str = BASE_USDC_ADDRESS


class TaskMarketTaskPreview(BaseModel):
    model_config = ConfigDict(frozen=True)

    description: str = Field(min_length=1, max_length=10_000)
    deliverables: list[str] = Field(min_length=1, max_length=20)
    reward_usdc: Decimal
    maximum_spend_usdc: Decimal
    deadline: datetime
    mode: TaskMarketMode
    tags: list[str] = Field(default_factory=list, max_length=10)
    network: TaskMarketNetwork = Field(default_factory=TaskMarketNetwork)
    fingerprint: str = ""

    @classmethod
    def build(
        cls,
        *,
        description: str,
        deliverables: list[str],
        reward_usdc: Decimal,
        maximum_spend_usdc: Decimal,
        deadline: datetime,
        mode: str,
        tags: list[str],
    ) -> "TaskMarketTaskPreview":
        preview = cls(
            description=description,
            deliverables=deliverables,
            reward_usdc=reward_usdc,
            maximum_spend_usdc=maximum_spend_usdc,
            deadline=deadline,
            mode=TaskMarketMode(mode),
            tags=tags,
        )
        return preview.model_copy(update={"fingerprint": preview.calculate_fingerprint()})

    def calculate_fingerprint(self) -> str:
        canonical = json.dumps(
            self._canonical_payload(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    def verify_fingerprint(self) -> None:
        if not self.fingerprint or self.fingerprint != self.calculate_fingerprint():
            raise ValueError("Task preview fingerprint does not match its contents")

    def full_description(self) -> str:
        items = "\n".join(f"- {item}" for item in self.deliverables)
        return f"{self.description.strip()}\n\nDeliverables:\n{items}"

    def remaining_duration_hours(self, now: datetime) -> Decimal:
        seconds = Decimal(str((self.deadline - now).total_seconds()))
        if seconds <= 0:
            raise ValueError("Task deadline must be in the future")
        return (seconds / Decimal(3600)).quantize(Decimal("0.000001"))

    @field_validator("reward_usdc", "maximum_spend_usdc", mode="before")
    @classmethod
    def validate_usdc(cls, value: Any) -> Decimal:
        try:
            raw_amount = Decimal(str(value))
            amount = raw_amount.quantize(USDC_QUANTUM)
        except (InvalidOperation, ValueError) as error:
            raise ValueError("USDC amount must have at most six decimals") from error
        if amount != raw_amount:
            raise ValueError("USDC amount must have at most six decimals")
        if not amount.is_finite() or amount <= 0:
            raise ValueError("USDC amount must be greater than zero")
        return amount

    @field_validator("deadline")
    @classmethod
    def validate_deadline(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Task deadline must include a timezone")
        return value.astimezone(timezone.utc)

    @field_validator("deliverables", "tags")
    @classmethod
    def validate_string_list(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values]
        if any(not value for value in cleaned):
            raise ValueError("List entries must not be empty")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("List entries must be unique")
        return cleaned

    @model_validator(mode="after")
    def validate_budget_and_network(self) -> "TaskMarketTaskPreview":
        if self.reward_usdc > self.maximum_spend_usdc:
            raise ValueError("Reward exceeds the declared maximum spend")
        if self.network.chain_id != BASE_CHAIN_ID:
            raise ValueError("TaskMarket funding is restricted to Base chain 8453")
        if self.network.usdc_contract.lower() != BASE_USDC_ADDRESS.lower():
            raise ValueError("TaskMarket funding requires canonical Base USDC")
        return self

    def _canonical_payload(self) -> dict[str, Any]:
        return {
            "deadline": self.deadline.isoformat(),
            "deliverables": self.deliverables,
            "description": self.description.strip(),
            "maximum_spend_usdc": format(self.maximum_spend_usdc, "f"),
            "mode": self.mode.value,
            "network": self.network.model_dump(),
            "reward_usdc": format(self.reward_usdc, "f"),
            "tags": self.tags,
        }


class TaskMarketPreflight(BaseModel):
    wallet_address: str
    chain_id: int
    usdc_contract: str
    balance_usdc: Decimal
    legal_accepted: bool


class TaskMarketCreationResult(BaseModel):
    task_id: str
    task_url: str
    live_status: dict[str, Any]
