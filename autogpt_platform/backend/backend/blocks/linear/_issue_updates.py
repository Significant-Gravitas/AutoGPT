from datetime import date
from enum import Enum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from backend.sdk import SchemaField


class ClearIssueField(str, Enum):
    ASSIGNEE = "assignee"
    DESCRIPTION = "description"
    DUE_DATE = "due_date"
    ESTIMATE = "estimate"


class IssueChanges(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = SchemaField(
        default=None, description="New title. Omit to leave unchanged.", min_length=1
    )
    description: str | None = SchemaField(
        default=None,
        description="New Markdown description. An empty string clears it; omit to leave unchanged.",
    )
    state_id: UUID | None = SchemaField(
        default=None,
        description="Workflow state UUID from the issue's team, not a state name.",
    )
    priority: int | None = SchemaField(
        default=None,
        ge=0,
        le=4,
        description="0: no priority, 1: urgent, 2: high, 3: medium, 4: low. Omit to leave unchanged.",
    )
    assignee_id: UUID | None = SchemaField(
        default=None, description="Assignee user UUID. Use clear_fields to unassign."
    )
    label_ids: list[UUID] | None = SchemaField(
        default=None,
        description="Replace all labels with these UUIDs. An empty list removes all labels. Do not combine with add/remove labels.",
    )
    add_label_ids: list[UUID] = SchemaField(
        default_factory=list,
        description="Label UUIDs to add atomically, preserving other labels.",
    )
    remove_label_ids: list[UUID] = SchemaField(
        default_factory=list,
        description="Label UUIDs to remove atomically, preserving other labels.",
    )
    due_date: date | None = SchemaField(
        default=None,
        description="Due date in YYYY-MM-DD format. Omit to leave unchanged.",
    )
    estimate: int | None = SchemaField(
        default=None,
        ge=0,
        description="Estimate points supported by the issue's team. Omit to leave unchanged.",
    )
    clear_fields: list[ClearIssueField] = SchemaField(
        default_factory=list,
        description="Explicitly clear these nullable fields. Other omitted or null inputs leave fields unchanged.",
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Title must not be blank")
        return value

    @model_validator(mode="after")
    def validate_changes(self) -> "IssueChanges":
        clear_values = {
            ClearIssueField.ASSIGNEE: self.assignee_id,
            ClearIssueField.DESCRIPTION: self.description,
            ClearIssueField.DUE_DATE: self.due_date,
            ClearIssueField.ESTIMATE: self.estimate,
        }
        if any(clear_values[field] is not None for field in self.clear_fields):
            raise ValueError("Cannot set and clear the same field")
        if self.label_ids is not None and (self.add_label_ids or self.remove_label_ids):
            raise ValueError("Cannot replace labels and add/remove labels together")
        if set(self.add_label_ids) & set(self.remove_label_ids):
            raise ValueError("Cannot add and remove the same label")
        if not self.to_api_input():
            raise ValueError("Provide at least one issue change")
        return self

    def to_api_input(self) -> dict[str, object]:
        field_names = {
            "title": "title",
            "description": "description",
            "state_id": "stateId",
            "priority": "priority",
            "assignee_id": "assigneeId",
            "label_ids": "labelIds",
            "due_date": "dueDate",
            "estimate": "estimate",
        }
        values = self.model_dump(mode="json", exclude_none=True)
        updates = {
            target: values[source]
            for source, target in field_names.items()
            if source in values
        }
        if self.add_label_ids:
            updates["addedLabelIds"] = values["add_label_ids"]
        if self.remove_label_ids:
            updates["removedLabelIds"] = values["remove_label_ids"]
        clear_names = {
            ClearIssueField.ASSIGNEE: "assigneeId",
            ClearIssueField.DESCRIPTION: "description",
            ClearIssueField.DUE_DATE: "dueDate",
            ClearIssueField.ESTIMATE: "estimate",
        }
        updates.update({clear_names[field]: None for field in self.clear_fields})
        return updates
