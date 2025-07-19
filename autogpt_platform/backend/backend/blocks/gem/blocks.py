"""
GEM API Blocks

This module implements blocks for interacting with the GEM recruiting platform API.
GEM provides programmatic access to manage candidates, projects, notes, and more.
"""

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from backend.sdk import (
    APIKeyCredentials,
    Block,
    BlockCategory,
    BlockOutput,
    BlockSchema,
    CredentialsMetaInput,
    Requests,
    SchemaField,
)

from ._config import gem


# Enums for GEM API
class PrivacyType(str, Enum):
    CONFIDENTIAL = "confidential"
    PERSONAL = "personal"
    SHARED = "shared"


class EventType(str, Enum):
    SEQUENCES = "sequences"
    SEQUENCE_REPLIES = "sequence_replies"
    MANUAL_TOUCHPOINTS = "manual_touchpoints"


class EventSubtype(str, Enum):
    FIRST_OUTREACH = "first_outreach"
    FOLLOW_UP = "follow_up"
    REPLY = "reply"


class ContactMedium(str, Enum):
    INMAIL = "inmail"
    PHONE_CALL = "phone_call"
    TEXT_MESSAGE = "text_message"
    EMAIL = "email"
    MEETING = "meeting"
    LI_CONNECT_REQUEST = "li_connect_request"


class ReplyStatus(str, Enum):
    INTERESTED = "interested"
    NOT_INTERESTED = "not_interested"
    LATER = "later"


class CustomFieldValueType(str, Enum):
    DATE = "date"
    TEXT = "text"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"


class CustomFieldScope(str, Enum):
    TEAM = "team"
    PROJECT = "project"


class ProjectFieldType(str, Enum):
    TEXT = "text"
    SINGLE_SELECT = "single_select"
    MULTI_SELECT = "multi_select"


class SourcedFrom(str, Enum):
    SEEKOUT = "SeekOut"
    HIREEZ = "hireEZ"
    STARCIRCLE = "Starcircle"
    CENSIA = "Censia"
    CONSIDER = "Consider"


# Base class for GEM blocks
class GemBlockBase(Block):
    """Base class for all GEM blocks with common functionality."""

    @staticmethod
    async def make_request(
        method: str,
        endpoint: str,
        credentials: APIKeyCredentials,
        data: Optional[dict] = None,
        params: Optional[dict] = None,
        files: Optional[dict] = None,
    ) -> dict:
        """Make an authenticated request to the GEM API."""
        api_key = credentials.api_key.get_secret_value()
        headers = {
            "X-API-Key": api_key,
            "Content-Type": "application/json" if not files else None,
        }

        # Remove Content-Type header if files are being uploaded
        if files and "Content-Type" in headers:
            del headers["Content-Type"]

        base_url = "https://api.gem.com"
        url = f"{base_url}{endpoint}"

        response = await Requests().request(
            method=method,
            url=url,
            headers=headers,
            json=data if not files else None,
            data=data if files else None,
            params=params,
            files=files,  # type: ignore
        )

        if response.status == 204:
            return {}

        if response.status < 200 or response.status >= 300:
            try:
                error_data = response.json()
            except Exception:
                error_data = {"message": response.text()}
            raise Exception(f"GEM API error ({response.status}): {error_data}")

        return response.json()


# User Management Blocks
class GemUserListBlock(GemBlockBase):
    """
    List all users on the GEM team.

    Returns a list of users with their ID, name, and email.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        email: Optional[str] = SchemaField(
            description="Filter by email address", default=None
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        users: List[Dict[str, Any]] = SchemaField(description="List of users")
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="4ba38bc0-e100-412a-92b9-c242c4308dbb",
            description="List all users on the GEM team",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params: Dict[str, Any] = {
            "page": input_data.page,
            "page_size": input_data.page_size,
        }
        if input_data.email:
            params["email"] = input_data.email

        users = await self.make_request(
            method="GET",
            endpoint="/v0/users",
            credentials=credentials,
            params=params,
        )

        # Extract pagination from response headers if available
        # Note: In a real implementation, we'd parse the X-Pagination header
        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
        }

        yield "users", users
        yield "pagination", pagination


# Candidate Management Blocks
class GemCandidateCreateBlock(GemBlockBase):
    """
    Create a new candidate in GEM.

    Supports de-duplication based on LinkedIn handle.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        created_by: str = SchemaField(description="User ID of the creator (required)")
        first_name: Optional[str] = SchemaField(
            description="Candidate's first name", default=None, max_length=255
        )
        last_name: Optional[str] = SchemaField(
            description="Candidate's last name", default=None, max_length=255
        )
        nickname: Optional[str] = SchemaField(
            description="Candidate's nickname", default=None, max_length=255
        )
        emails: List[Dict[str, Any]] = SchemaField(
            description="List of email objects", default_factory=list
        )
        linked_in_handle: Optional[str] = SchemaField(
            description="LinkedIn handle (used for de-duplication)",
            default=None,
            max_length=255,
        )
        title: Optional[str] = SchemaField(
            description="Job title", default=None, max_length=255
        )
        company: Optional[str] = SchemaField(
            description="Current company", default=None, max_length=255
        )
        location: Optional[str] = SchemaField(
            description="Location", default=None, max_length=255
        )
        school: Optional[str] = SchemaField(
            description="School/University", default=None, max_length=255
        )
        phone_number: Optional[str] = SchemaField(
            description="Phone number", default=None, max_length=255
        )
        project_ids: List[str] = SchemaField(
            description="Project IDs to add candidate to", default_factory=list
        )
        sourced_from: Optional[SourcedFrom] = SchemaField(
            description="Source platform", default=None
        )
        autofill: bool = SchemaField(
            description="Attempt to fill missing fields (requires LinkedIn handle)",
            default=False,
        )

    class Output(BlockSchema):
        candidate: Dict[str, Any] = SchemaField(description="Created candidate object")
        duplicate_found: bool = SchemaField(
            description="Whether a duplicate was found", default=False
        )
        duplicate_candidate: Optional[Dict[str, Any]] = SchemaField(
            description="Duplicate candidate info if found", default=None
        )

    def __init__(self):
        super().__init__(
            id="48b25bdd-c386-4dc6-9457-0f49137e0476",
            description="Create a new candidate in GEM",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {
            "created_by": input_data.created_by,
            "autofill": input_data.autofill,
        }

        # Add optional fields
        optional_fields = [
            "first_name",
            "last_name",
            "nickname",
            "linked_in_handle",
            "title",
            "company",
            "location",
            "school",
            "phone_number",
        ]
        for field in optional_fields:
            value = getattr(input_data, field)
            if value is not None:
                data[field] = value

        # Add lists
        if input_data.emails:
            data["emails"] = input_data.emails
        if input_data.project_ids:
            data["project_ids"] = input_data.project_ids
        if input_data.sourced_from:
            data["sourced_from"] = input_data.sourced_from

        try:
            candidate = await self.make_request(
                method="POST",
                endpoint="/v0/candidates",
                credentials=credentials,
                data=data,
            )
            yield "candidate", candidate
            yield "duplicate_found", False
            yield "duplicate_candidate", None
        except Exception as e:
            # Check if it's a duplicate error
            if "duplicate_candidate" in str(e):
                # Parse duplicate info from error
                yield "candidate", {}
                yield "duplicate_found", True
                yield "duplicate_candidate", {"error": str(e)}
            else:
                raise


class GemCandidateGetBlock(GemBlockBase):
    """
    Fetch a candidate by ID from GEM.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID to fetch")

    class Output(BlockSchema):
        candidate: Dict[str, Any] = SchemaField(description="Candidate object")

    def __init__(self):
        super().__init__(
            id="2d91089d-edc5-40c4-84f9-99a9e66260b7",
            description="Fetch a candidate by ID",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        candidate = await self.make_request(
            method="GET",
            endpoint=f"/v0/candidates/{input_data.candidate_id}",
            credentials=credentials,
        )
        yield "candidate", candidate


class GemCandidateListBlock(GemBlockBase):
    """
    List and search candidates in GEM.

    Supports various filters and pagination.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        email: Optional[str] = SchemaField(
            description="Filter by email (partial match)", default=None, max_length=255
        )
        linked_in_handle: Optional[str] = SchemaField(
            description="Filter by LinkedIn handle", default=None, max_length=255
        )
        created_by: Optional[str] = SchemaField(
            description="Filter by creator user ID", default=None
        )
        created_after: Optional[int] = SchemaField(
            description="Unix timestamp - only candidates created after",
            default=None,
            ge=1,
        )
        created_before: Optional[int] = SchemaField(
            description="Unix timestamp - only candidates created before",
            default=None,
            ge=1,
        )
        updated_after: Optional[int] = SchemaField(
            description="Unix timestamp - only candidates updated after",
            default=None,
            ge=1,
        )
        candidate_ids: List[str] = SchemaField(
            description="Filter by specific candidate IDs (max 20)",
            default_factory=list,
        )
        sort: Literal["asc", "desc"] = SchemaField(
            description="Sort order", default="desc"
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        candidates: List[Dict[str, Any]] = SchemaField(description="List of candidates")
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="c2c08e62-b74e-4429-8d86-5ede6c109faf",
            description="List and search candidates",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        # Add optional filters
        if input_data.email:
            params["email"] = input_data.email
        if input_data.linked_in_handle:
            params["linked_in_handle"] = input_data.linked_in_handle
        if input_data.created_by:
            params["created_by"] = input_data.created_by
        if input_data.created_after:
            params["created_after"] = input_data.created_after
        if input_data.created_before:
            params["created_before"] = input_data.created_before
        if input_data.updated_after:
            params["updated_after"] = input_data.updated_after
        if input_data.candidate_ids:
            params["candidate_ids"] = ",".join(input_data.candidate_ids[:20])

        candidates = await self.make_request(
            method="GET",
            endpoint="/v0/candidates",
            credentials=credentials,
            params=params,
        )

        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        yield "candidates", candidates
        yield "pagination", pagination


class GemCandidateUpdateBlock(GemBlockBase):
    """
    Update a candidate's information in GEM.

    Only specified fields will be updated.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID to update")
        first_name: Optional[str] = SchemaField(
            description="Update first name", default=None, max_length=255
        )
        last_name: Optional[str] = SchemaField(
            description="Update last name", default=None, max_length=255
        )
        nickname: Optional[str] = SchemaField(
            description="Update nickname", default=None, max_length=255
        )
        emails: Optional[List[Dict[str, Any]]] = SchemaField(
            description="Update email list", default=None
        )
        title: Optional[str] = SchemaField(
            description="Update job title", default=None, max_length=255
        )
        company: Optional[str] = SchemaField(
            description="Update company", default=None, max_length=255
        )
        location: Optional[str] = SchemaField(
            description="Update location", default=None, max_length=255
        )
        school: Optional[str] = SchemaField(
            description="Update school", default=None, max_length=255
        )
        phone_number: Optional[str] = SchemaField(
            description="Update phone number", default=None, max_length=255
        )

    class Output(BlockSchema):
        candidate: Dict[str, Any] = SchemaField(description="Updated candidate object")

    def __init__(self):
        super().__init__(
            id="c91b5257-d0da-4d74-b297-fbbd2ef4f8d0",
            description="Update a candidate's information",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {}

        # Add fields to update
        update_fields = [
            "first_name",
            "last_name",
            "nickname",
            "emails",
            "title",
            "company",
            "location",
            "school",
            "phone_number",
        ]
        for field in update_fields:
            value = getattr(input_data, field)
            if value is not None:
                data[field] = value

        if not data:
            raise ValueError("No fields to update")

        candidate = await self.make_request(
            method="PUT",
            endpoint=f"/v0/candidates/{input_data.candidate_id}",
            credentials=credentials,
            data=data,
        )

        yield "candidate", candidate


class GemCandidateDeleteBlock(GemBlockBase):
    """
    Delete a candidate from GEM.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID to delete")
        on_behalf_of_user_id: str = SchemaField(
            description="User ID performing the deletion"
        )
        permanently_remove_contact_info: bool = SchemaField(
            description="Prevent contact info from being re-added", default=False
        )

    class Output(BlockSchema):
        success: bool = SchemaField(description="Whether deletion was successful")

    def __init__(self):
        super().__init__(
            id="be9a19d4-9158-4cfe-a133-ea4ff1932735",
            description="Delete a candidate",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {
            "on_behalf_of_user_id": input_data.on_behalf_of_user_id,
            "permanently_remove_contact_info": input_data.permanently_remove_contact_info,
        }

        await self.make_request(
            method="DELETE",
            endpoint=f"/v0/candidates/{input_data.candidate_id}",
            credentials=credentials,
            data=data,
        )

        yield "success", True


# Event Management Blocks
class GemEventCreateBlock(GemBlockBase):
    """
    Create a manual touchpoint event for a candidate.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID")
        timestamp: int = SchemaField(description="Unix timestamp of the event", ge=1)
        user_id: str = SchemaField(description="User ID who created the event")
        on_behalf_of_user_id: Optional[str] = SchemaField(
            description="User ID on whose behalf the event was created", default=None
        )
        project_id: str = SchemaField(description="Project ID")
        sequence_id: str = SchemaField(description="Sequence ID")
        type: EventType = SchemaField(
            description="Event type", default=EventType.MANUAL_TOUCHPOINTS
        )
        subtype: EventSubtype = SchemaField(description="Event subtype")
        contact_medium: ContactMedium = SchemaField(description="Contact medium used")
        reply_status: Optional[ReplyStatus] = SchemaField(
            description="Reply status", default=None
        )

    class Output(BlockSchema):
        event: Dict[str, Any] = SchemaField(description="Created event object")

    def __init__(self):
        super().__init__(
            id="a2a8e3f8-a3a9-4c54-957e-74ffe37b97a2",
            description="Create a manual touchpoint event",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {
            "timestamp": input_data.timestamp,
            "user_id": input_data.user_id,
            "project_id": input_data.project_id,
            "sequence_id": input_data.sequence_id,
            "type": input_data.type,
            "subtype": input_data.subtype,
            "contact_medium": input_data.contact_medium,
        }

        if input_data.on_behalf_of_user_id:
            data["on_behalf_of_user_id"] = input_data.on_behalf_of_user_id
        if input_data.reply_status:
            data["reply_status"] = input_data.reply_status

        event = await self.make_request(
            method="POST",
            endpoint=f"/v0/candidates/{input_data.candidate_id}/events",
            credentials=credentials,
            data=data,
        )

        yield "event", event


class GemEventListBlock(GemBlockBase):
    """
    List events for a candidate.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID")
        created_after: Optional[int] = SchemaField(
            description="Unix timestamp - only events created after", default=None, ge=1
        )
        created_before: Optional[int] = SchemaField(
            description="Unix timestamp - only events created before",
            default=None,
            ge=1,
        )
        sort: Literal["asc", "desc"] = SchemaField(
            description="Sort order", default="desc"
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        events: List[Dict[str, Any]] = SchemaField(description="List of events")
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="94fab978-947a-489e-91ed-de630094d13e",
            description="List events for a candidate",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        if input_data.created_after:
            params["created_after"] = input_data.created_after
        if input_data.created_before:
            params["created_before"] = input_data.created_before

        events = await self.make_request(
            method="GET",
            endpoint=f"/v0/candidates/{input_data.candidate_id}/events",
            credentials=credentials,
            params=params,
        )

        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        yield "events", events
        yield "pagination", pagination


# Note Management Blocks
class GemNoteCreateBlock(GemBlockBase):
    """
    Create a note for a candidate.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID")
        user_id: str = SchemaField(description="User ID creating the note")
        content: str = SchemaField(description="Note content", max_length=10000)
        is_private: bool = SchemaField(
            description="Whether the note is private", default=False
        )

    class Output(BlockSchema):
        note: Dict[str, Any] = SchemaField(description="Created note object")

    def __init__(self):
        super().__init__(
            id="4c06e1b1-b1e7-498a-915d-a7921da05488",
            description="Create a note for a candidate",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {
            "candidate_id": input_data.candidate_id,
            "user_id": input_data.user_id,
            "content": input_data.content,
            "is_private": input_data.is_private,
        }

        note = await self.make_request(
            method="POST",
            endpoint="/v0/notes",
            credentials=credentials,
            data=data,
        )

        yield "note", note


class GemNoteListBlock(GemBlockBase):
    """
    List notes for a candidate.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID")
        created_after: Optional[int] = SchemaField(
            description="Unix timestamp - only notes created after", default=None, ge=1
        )
        created_before: Optional[int] = SchemaField(
            description="Unix timestamp - only notes created before", default=None, ge=1
        )
        sort: Literal["asc", "desc"] = SchemaField(
            description="Sort order", default="desc"
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        notes: List[Dict[str, Any]] = SchemaField(description="List of notes")
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="4aed2164-78e2-4fda-b1be-606075cc163b",
            description="List notes for a candidate",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        if input_data.created_after:
            params["created_after"] = input_data.created_after
        if input_data.created_before:
            params["created_before"] = input_data.created_before

        notes = await self.make_request(
            method="GET",
            endpoint=f"/v0/candidates/{input_data.candidate_id}/notes",
            credentials=credentials,
            params=params,
        )

        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        yield "notes", notes
        yield "pagination", pagination


# Resume Management Block
class GemResumeUploadBlock(GemBlockBase):
    """
    Upload a resume for a candidate.

    Supports PDF, DOC, and DOCX formats up to 10MB.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        candidate_id: str = SchemaField(description="The candidate ID")
        user_id: str = SchemaField(description="User ID uploading the resume")
        file_content: bytes = SchemaField(
            description="Resume file content (PDF, DOC, or DOCX)",
            exclude=True,  # Don't include in JSON serialization
        )
        filename: str = SchemaField(description="Resume filename", default="resume.pdf")

    class Output(BlockSchema):
        resume: Dict[str, Any] = SchemaField(
            description="Uploaded resume object with download URL"
        )

    def __init__(self):
        super().__init__(
            id="0a57b239-77f4-4e12-9039-32948fc1dc83",
            description="Upload a resume for a candidate",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        files = {"resume_file": (input_data.filename, input_data.file_content)}

        resume = await self.make_request(
            method="POST",
            endpoint=f"/v0/candidates/{input_data.candidate_id}/uploaded_resumes/{input_data.user_id}",
            credentials=credentials,
            files=files,
        )

        yield "resume", resume


# Project Management Blocks
class GemProjectCreateBlock(GemBlockBase):
    """
    Create a new project in GEM.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        user_id: str = SchemaField(description="Project owner user ID")
        name: str = SchemaField(description="Project name", max_length=255)
        privacy_type: PrivacyType = SchemaField(
            description="Project privacy type", default=PrivacyType.PERSONAL
        )
        description: Optional[str] = SchemaField(
            description="Project description", default=None, max_length=2000
        )

    class Output(BlockSchema):
        project: Dict[str, Any] = SchemaField(description="Created project object")

    def __init__(self):
        super().__init__(
            id="d33f2c6e-4b2c-4d49-98c8-351ebb884034",
            description="Create a new project",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {
            "user_id": input_data.user_id,
            "name": input_data.name,
            "privacy_type": input_data.privacy_type,
        }

        if input_data.description:
            data["description"] = input_data.description

        project = await self.make_request(
            method="POST",
            endpoint="/v0/projects",
            credentials=credentials,
            data=data,
        )

        yield "project", project


class GemProjectGetBlock(GemBlockBase):
    """
    Get a project by ID.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        project_id: str = SchemaField(description="The project ID to fetch")

    class Output(BlockSchema):
        project: Dict[str, Any] = SchemaField(description="Project object")

    def __init__(self):
        super().__init__(
            id="3cf28c29-6664-44aa-afb8-edeff58ab123",
            description="Get a project by ID",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        project = await self.make_request(
            method="GET",
            endpoint=f"/v0/projects/{input_data.project_id}",
            credentials=credentials,
        )

        yield "project", project


class GemProjectListBlock(GemBlockBase):
    """
    List projects with various filters.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        user_id: Optional[str] = SchemaField(
            description="Filter by owner user ID", default=None
        )
        readable_by: Optional[str] = SchemaField(
            description="Filter by user with read access", default=None
        )
        writable_by: Optional[str] = SchemaField(
            description="Filter by user with write access", default=None
        )
        is_archived: Optional[bool] = SchemaField(
            description="Filter by archived status", default=None
        )
        created_after: Optional[int] = SchemaField(
            description="Unix timestamp - only projects created after",
            default=None,
            ge=1,
        )
        created_before: Optional[int] = SchemaField(
            description="Unix timestamp - only projects created before",
            default=None,
            ge=1,
        )
        sort: Literal["asc", "desc"] = SchemaField(
            description="Sort order", default="desc"
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        projects: List[Dict[str, Any]] = SchemaField(description="List of projects")
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="7c9ac248-b6f9-47ad-82e2-86605e9a3e0e",
            description="List projects with filters",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        # Add optional filters
        if input_data.user_id:
            params["user_id"] = input_data.user_id
        if input_data.readable_by:
            params["readable_by"] = input_data.readable_by
        if input_data.writable_by:
            params["writable_by"] = input_data.writable_by
        if input_data.is_archived is not None:
            params["is_archived"] = input_data.is_archived
        if input_data.created_after:
            params["created_after"] = input_data.created_after
        if input_data.created_before:
            params["created_before"] = input_data.created_before

        projects = await self.make_request(
            method="GET",
            endpoint="/v0/projects",
            credentials=credentials,
            params=params,
        )

        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        yield "projects", projects
        yield "pagination", pagination


class GemProjectUpdateBlock(GemBlockBase):
    """
    Update a project's information.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        project_id: str = SchemaField(description="The project ID to update")
        user_id: Optional[str] = SchemaField(
            description="Update project owner", default=None
        )
        name: Optional[str] = SchemaField(
            description="Update project name", default=None, max_length=255
        )
        privacy_type: Optional[PrivacyType] = SchemaField(
            description="Update privacy type", default=None
        )
        description: Optional[str] = SchemaField(
            description="Update description", default=None, max_length=2000
        )
        is_archived: Optional[bool] = SchemaField(
            description="Update archived status", default=None
        )

    class Output(BlockSchema):
        project: Dict[str, Any] = SchemaField(description="Updated project object")

    def __init__(self):
        super().__init__(
            id="e8d54fc3-e305-4586-9354-15ecad1fc663",
            description="Update a project",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {}

        # Add fields to update
        if input_data.user_id:
            data["user_id"] = input_data.user_id
        if input_data.name:
            data["name"] = input_data.name
        if input_data.privacy_type:
            data["privacy_type"] = input_data.privacy_type
        if input_data.description is not None:
            data["description"] = input_data.description
        if input_data.is_archived is not None:
            data["is_archived"] = input_data.is_archived

        if not data:
            raise ValueError("No fields to update")

        project = await self.make_request(
            method="PATCH",
            endpoint=f"/v0/projects/{input_data.project_id}",
            credentials=credentials,
            data=data,
        )

        yield "project", project


class GemProjectAddCandidatesBlock(GemBlockBase):
    """
    Add candidates to a project.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        project_id: str = SchemaField(description="The project ID")
        candidate_ids: List[str] = SchemaField(
            description="List of candidate IDs to add (max 1000)"
        )
        user_id: Optional[str] = SchemaField(
            description="User performing the update (must have write access)",
            default=None,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether candidates were added successfully"
        )
        already_in_project: List[str] = SchemaField(
            description="Candidate IDs already in the project", default_factory=list
        )

    def __init__(self):
        super().__init__(
            id="54917c9d-4842-4e6e-829f-b11de520d6b6",
            description="Add candidates to a project",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data: Dict[str, Any] = {
            "candidate_ids": input_data.candidate_ids[:1000],
        }

        if input_data.user_id:
            data["user_id"] = input_data.user_id

        try:
            await self.make_request(
                method="PUT",
                endpoint=f"/v0/projects/{input_data.project_id}/candidates",
                credentials=credentials,
                data=data,
            )
            yield "success", True
            yield "already_in_project", []
        except Exception as e:
            # Parse error for already existing candidates
            if "already in the project" in str(e):
                yield "success", False
                yield "already_in_project", input_data.candidate_ids
            else:
                raise


class GemProjectRemoveCandidatesBlock(GemBlockBase):
    """
    Remove candidates from a project.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        project_id: str = SchemaField(description="The project ID")
        candidate_ids: List[str] = SchemaField(
            description="List of candidate IDs to remove (max 1000)"
        )
        user_id: Optional[str] = SchemaField(
            description="User performing the update (must have write access)",
            default=None,
        )

    class Output(BlockSchema):
        success: bool = SchemaField(
            description="Whether candidates were removed successfully"
        )
        not_in_project: List[str] = SchemaField(
            description="Candidate IDs not in the project", default_factory=list
        )

    def __init__(self):
        super().__init__(
            id="1e06b2f1-eb53-4cff-995d-0ddd123bff06",
            description="Remove candidates from a project",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data: Dict[str, Any] = {
            "candidate_ids": input_data.candidate_ids[:1000],
        }

        if input_data.user_id:
            data["user_id"] = input_data.user_id

        try:
            await self.make_request(
                method="DELETE",
                endpoint=f"/v0/projects/{input_data.project_id}/candidates",
                credentials=credentials,
                data=data,
            )
            yield "success", True
            yield "not_in_project", []
        except Exception as e:
            # Parse error for candidates not in project
            if "not in the project" in str(e):
                yield "success", False
                yield "not_in_project", input_data.candidate_ids
            else:
                raise


# Custom Field Blocks
class GemCustomFieldListBlock(GemBlockBase):
    """
    List custom fields.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        scope: Optional[CustomFieldScope] = SchemaField(
            description="Filter by scope (team or project)", default=None
        )
        project_id: Optional[str] = SchemaField(
            description="Project ID (required when scope is project)", default=None
        )
        is_hidden: Optional[bool] = SchemaField(
            description="Filter by hidden status", default=None
        )
        name: Optional[str] = SchemaField(description="Filter by name", default=None)
        created_after: Optional[int] = SchemaField(
            description="Unix timestamp - only fields created after", default=None, ge=1
        )
        created_before: Optional[int] = SchemaField(
            description="Unix timestamp - only fields created before",
            default=None,
            ge=1,
        )
        sort: Literal["asc", "desc"] = SchemaField(
            description="Sort order", default="desc"
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        custom_fields: List[Dict[str, Any]] = SchemaField(
            description="List of custom fields"
        )
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="cf4e21fb-75d5-4b1c-87d7-1379eaddec1b",
            description="List custom fields",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        # Add optional filters
        if input_data.scope:
            params["scope"] = input_data.scope
        if input_data.project_id:
            params["project_id"] = input_data.project_id
        if input_data.is_hidden is not None:
            params["is_hidden"] = input_data.is_hidden
        if input_data.name:
            params["name"] = input_data.name
        if input_data.created_after:
            params["created_after"] = input_data.created_after
        if input_data.created_before:
            params["created_before"] = input_data.created_before

        custom_fields = await self.make_request(
            method="GET",
            endpoint="/v0/custom_fields",
            credentials=credentials,
            params=params,
        )

        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        yield "custom_fields", custom_fields
        yield "pagination", pagination


class GemCustomFieldCreateBlock(GemBlockBase):
    """
    Create a custom field.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        name: str = SchemaField(
            description="Custom field name (unique in scope)",
            min_length=1,
            max_length=50,
        )
        value_type: CustomFieldValueType = SchemaField(description="Field value type")
        scope: CustomFieldScope = SchemaField(
            description="Field scope (team or project)"
        )
        project_id: Optional[str] = SchemaField(
            description="Project ID (required when scope is project)", default=None
        )
        option_values: List[str] = SchemaField(
            description="Options for single_select/multi_select fields",
            default_factory=list,
        )

    class Output(BlockSchema):
        custom_field: Dict[str, Any] = SchemaField(
            description="Created custom field object"
        )
        duplicate_found: bool = SchemaField(
            description="Whether a duplicate was found", default=False
        )
        duplicate_field: Optional[Dict[str, Any]] = SchemaField(
            description="Duplicate field info if found", default=None
        )

    def __init__(self):
        super().__init__(
            id="daa6170a-cc5f-4353-82e1-d27c6c77832d",
            description="Create a custom field",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        data = {
            "name": input_data.name.strip(),
            "value_type": input_data.value_type,
            "scope": input_data.scope,
        }

        if input_data.scope == CustomFieldScope.PROJECT:
            if not input_data.project_id:
                raise ValueError("project_id is required when scope is project")
            data["project_id"] = input_data.project_id

        if input_data.value_type in [
            CustomFieldValueType.SINGLE_SELECT,
            CustomFieldValueType.MULTI_SELECT,
        ]:
            if not input_data.option_values:
                raise ValueError("option_values required for select fields")
            data["option_values"] = input_data.option_values

        try:
            custom_field = await self.make_request(
                method="POST",
                endpoint="/v0/custom_fields",
                credentials=credentials,
                data=data,
            )
            yield "custom_field", custom_field
            yield "duplicate_found", False
            yield "duplicate_field", None
        except Exception as e:
            # Check if it's a duplicate error
            if "already existed" in str(e):
                yield "custom_field", {}
                yield "duplicate_found", True
                yield "duplicate_field", {"error": str(e)}
            else:
                raise


class GemCustomFieldOptionAddBlock(GemBlockBase):
    """
    Add options to a single_select or multi_select custom field.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        custom_field_id: str = SchemaField(description="The custom field ID")
        option_values: List[str] = SchemaField(
            description="New option values to add", min_length=1
        )

    class Output(BlockSchema):
        options: List[Dict[str, Any]] = SchemaField(description="Created options")
        duplicate_options: List[Dict[str, Any]] = SchemaField(
            description="Options that already existed", default_factory=list
        )

    def __init__(self):
        super().__init__(
            id="d5b4376d-93bf-47af-9559-00bebbcc11cf",
            description="Add options to a custom field",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        # Clean option values
        clean_values = [v.strip() for v in input_data.option_values if v.strip()]

        data = {"option_values": clean_values}

        try:
            options = await self.make_request(
                method="POST",
                endpoint=f"/v0/custom_fields/{input_data.custom_field_id}/options",
                credentials=credentials,
                data=data,
            )
            yield "options", options
            yield "duplicate_options", []
        except Exception as e:
            # Check if it's a duplicate error
            if "already existed" in str(e):
                yield "options", []
                yield "duplicate_options", clean_values
            else:
                raise


# Sequence Blocks
class GemSequenceListBlock(GemBlockBase):
    """
    List sequences.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )
        user_id: Optional[str] = SchemaField(
            description="Filter by user ID", default=None
        )
        created_after: Optional[int] = SchemaField(
            description="Unix timestamp - only sequences created after",
            default=None,
            ge=1,
        )
        created_before: Optional[int] = SchemaField(
            description="Unix timestamp - only sequences created before",
            default=None,
            ge=1,
        )
        sort: Literal["asc", "desc"] = SchemaField(
            description="Sort order", default="desc"
        )
        page: int = SchemaField(description="Page number (1-indexed)", default=1, ge=1)
        page_size: int = SchemaField(
            description="Number of results per page", default=20, ge=1, le=100
        )

    class Output(BlockSchema):
        sequences: List[Dict[str, Any]] = SchemaField(description="List of sequences")
        pagination: Dict[str, Any] = SchemaField(description="Pagination metadata")

    def __init__(self):
        super().__init__(
            id="83a74e30-ac63-4693-a778-c3e0cf4047b0",
            description="List sequences",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        params = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        # Add optional filters
        if input_data.user_id:
            params["user_id"] = input_data.user_id
        if input_data.created_after:
            params["created_after"] = input_data.created_after
        if input_data.created_before:
            params["created_before"] = input_data.created_before

        sequences = await self.make_request(
            method="GET",
            endpoint="/v0/sequences",
            credentials=credentials,
            params=params,
        )

        pagination = {
            "page": input_data.page,
            "page_size": input_data.page_size,
            "sort": input_data.sort,
        }

        yield "sequences", sequences
        yield "pagination", pagination


# Data Export Block
class GemDataExportGetBlock(GemBlockBase):
    """
    Get the most recent data export.
    """

    class Input(BlockSchema):
        credentials: CredentialsMetaInput = gem.credentials_field(
            description="GEM API credentials"
        )

    class Output(BlockSchema):
        export: Dict[str, Any] = SchemaField(
            description="Data export information with file links"
        )

    def __init__(self):
        super().__init__(
            id="f6141cbb-dbb3-4aae-9546-3a5dfe150f34",
            description="Get the most recent data export",
            categories={BlockCategory.CRM},
            input_schema=self.Input,
            output_schema=self.Output,
        )

    async def run(
        self, input_data: Input, *, credentials: APIKeyCredentials, **kwargs
    ) -> BlockOutput:
        export = await self.make_request(
            method="GET",
            endpoint="/v0/data_export",
            credentials=credentials,
        )

        yield "export", export
