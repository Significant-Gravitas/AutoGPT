"""Pydantic request/response models for organization management."""

from datetime import datetime
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

from backend.util.text_limits import (
    MAX_DESCRIPTION_LENGTH,
    MAX_EMAIL_LENGTH,
    MAX_ID_LENGTH,
    MAX_LIST_ENTRIES,
    MAX_NAME_LENGTH,
    MAX_URL_LENGTH,
    IDText,
)

# Hosts an org avatar URL may point at. Must stay in step with `images` in
# autogpt_platform/frontend/next.config.mjs (domains + remotePatterns) —
# guarded by test_avatar_allowlist_matches_frontend_image_config: next/image
# matches the configured protocol exactly, so next/image errors at render
# time for hosts outside that list.
# storage.googleapis.com is our own media storage — the host uploaded
# images live on — so an avatar uploaded like other images keeps working.
ALLOWED_AVATAR_HOSTS = frozenset(
    {
        "images.unsplash.com",
        "ddz4ak4pa3d19.cloudfront.net",
        "upload.wikimedia.org",
        "storage.googleapis.com",
        "ideogram.ai",
        "example.com",
        "storage.cloud.google.com",
        "lh3.googleusercontent.com",
    }
)

# Hosts the frontend only whitelists over https (in remotePatterns but not in
# `domains`) — an http:// avatar there would fail next/image at render time.
HTTPS_ONLY_AVATAR_HOSTS = frozenset(
    {
        "storage.cloud.google.com",
        "lh3.googleusercontent.com",
    }
)


class CreateOrgRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=MAX_NAME_LENGTH)
    slug: str = Field(
        ...,
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LENGTH)


class UpdateOrgRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=MAX_NAME_LENGTH)
    slug: str | None = Field(
        None,
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )
    description: str | None = Field(default=None, max_length=MAX_DESCRIPTION_LENGTH)
    avatar_url: str | None = Field(default=None, max_length=MAX_URL_LENGTH)

    @field_validator("avatar_url")
    @classmethod
    def _avatar_url_must_use_an_allowed_host(cls, value: str | None) -> str | None:
        """Reject avatar URLs the account menu cannot render safely.

        The menu renders the avatar through optimized ``next/image``, which
        errors at render time for hosts outside ``images`` in
        ``next.config.mjs``; serving the URL as a plain ``<img>`` instead
        would let the avatar's host see every member's IP address.
        Only unambiguous same-origin paths pass as relative: browsers fold
        ``\\`` into ``/`` inside http(s) URLs (so ``/\\host/x`` and
        ``\\\\host/x`` resolve to ``host``), and ``//host/x`` is a
        protocol-relative URL next/image refuses to optimize.
        """
        if value is None:
            return value

        # urlparse (like browser URL parsing) strips tabs, CR and LF from
        # anywhere in a URL, so "/\t/attacker.example/x" silently becomes the
        # protocol-relative "//attacker.example/x". Control characters have no
        # business in a path, so reject them before any classification.
        if "\t" in value or "\r" in value or "\n" in value:
            raise ValueError(
                "avatar_url must not contain tab, carriage-return or newline "
                "characters; URL parsers strip them, which can turn the value "
                "into a different URL than the one typed"
            )
        # Browsers treat "\" as "/" in http(s) URLs, so "\/host/x" and
        # "\\host/x" parse as relative paths here but resolve to an
        # attacker-chosen host; the backslash itself disqualifies the value.
        if "\\" in value:
            raise ValueError(
                "avatar_url must not contain backslashes; browsers read them "
                "as '/' in http(s) URLs, which turns relative paths into "
                "network paths"
            )
        # A leading "//" is a protocol-relative URL, not a local path.
        if value.startswith("//"):
            raise ValueError(
                "avatar_url must not be a protocol-relative URL ('//host/...'); "
                "use an absolute http(s) URL or a path starting with a single '/'"
            )

        parsed = urlparse(value)
        if parsed.scheme in ("http", "https"):
            if parsed.hostname not in ALLOWED_AVATAR_HOSTS:
                raise ValueError(
                    "avatar_url must be an http(s) URL on an allowed media host "
                    f"({', '.join(sorted(ALLOWED_AVATAR_HOSTS))}) or a path "
                    "starting with a single '/'"
                )
            if parsed.username is not None or parsed.password is not None:
                # userinfo has no legitimate use in an avatar URL, and
                # next/image's optimizer fetches the URL server-side, where
                # fetch() rejects http(s) URLs that embed credentials
                # (WHATWG fetch: "url includes credentials" -> TypeError),
                # so the avatar would silently fall back to the initial.
                raise ValueError(
                    "avatar_url must not embed credentials (user:password@host)"
                )
            # urlparse validates the port lazily; touching it rejects
            # "https://host:invalid/..." before the URL gets persisted.
            try:
                parsed.port  # noqa: B018 — attribute access is the validation
            except ValueError as exc:
                raise ValueError(f"avatar_url has an invalid port: {exc}") from exc
            if parsed.scheme == "http" and parsed.hostname in HTTPS_ONLY_AVATAR_HOSTS:
                raise ValueError(
                    f"avatar_url host {parsed.hostname} is only allowed over https"
                )
            return value
        if parsed.scheme:
            # javascript:, data:, ftp:, ... — not a renderable image URL
            raise ValueError(
                "avatar_url must be an http(s) URL or a path starting with a single '/'"
            )
        if not value.startswith("/"):
            # e.g. "a.png" — resolves against the current page's path
            # (e.g. /settings/...), not the media prefix
            raise ValueError(
                "avatar_url must be an absolute http(s) URL on an allowed media "
                "host or a path starting with a single '/'"
            )
        # an unambiguous same-origin path — renders from our own origin
        return value


class UpdateOrgData(BaseModel):
    """Structured data object for update_org DB function.

    Only these fields can be updated — no arbitrary dict keys.
    """

    name: str | None = None
    slug: str | None = None
    description: str | None = None
    avatar_url: str | None = None


class OrgResponse(BaseModel):
    id: str
    name: str
    slug: str
    avatar_url: str | None
    description: str | None
    is_personal: bool
    member_count: int
    created_at: datetime

    @staticmethod
    def from_db(org, member_count: int = 0) -> "OrgResponse":
        return OrgResponse(
            id=org.id,
            name=org.name,
            slug=org.slug,
            avatar_url=org.avatarUrl,
            description=org.description,
            is_personal=org.isPersonal,
            member_count=member_count,
            created_at=org.createdAt,
        )


class OrgMemberResponse(BaseModel):
    id: str
    user_id: str
    email: str
    name: str | None
    is_owner: bool
    is_admin: bool
    is_billing_manager: bool
    joined_at: datetime

    @staticmethod
    def from_db(member) -> "OrgMemberResponse":
        return OrgMemberResponse(
            id=member.id,
            user_id=member.userId,
            email=member.User.email if member.User else "",
            name=member.User.name if member.User else None,
            is_owner=member.isOwner,
            is_admin=member.isAdmin,
            is_billing_manager=member.isBillingManager,
            joined_at=member.joinedAt,
        )


class UpdateMemberRequest(BaseModel):
    is_admin: bool | None = None
    is_billing_manager: bool | None = None


class TransferOwnershipRequest(BaseModel):
    new_owner_id: str = Field(..., max_length=MAX_ID_LENGTH)


class OrgAliasResponse(BaseModel):
    id: str
    alias_slug: str
    alias_type: str
    created_at: datetime

    @staticmethod
    def from_db(alias) -> "OrgAliasResponse":
        return OrgAliasResponse(
            id=alias.id,
            alias_slug=alias.aliasSlug,
            alias_type=alias.aliasType,
            created_at=alias.createdAt,
        )


class CreateAliasRequest(BaseModel):
    alias_slug: str = Field(
        ...,
        min_length=1,
        max_length=MAX_NAME_LENGTH,
        pattern=r"^[a-z0-9][a-z0-9-]*$",
    )


class CreateInvitationRequest(BaseModel):
    email: str = Field(..., max_length=MAX_EMAIL_LENGTH)
    is_admin: bool = False
    is_billing_manager: bool = False
    team_ids: list[IDText] = Field(default_factory=list, max_length=MAX_LIST_ENTRIES)

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, email: str) -> str:
        # Stored lowercase so the invitee's pending list can match it exactly.
        email = email.strip().lower()
        if not email:
            raise ValueError("email must not be blank")
        return email


class InvitationResponse(BaseModel):
    id: str
    email: str
    is_admin: bool
    is_billing_manager: bool
    # token intentionally excluded — only returned on creation
    expires_at: datetime
    created_at: datetime
    team_ids: list[str]

    @staticmethod
    def from_db(inv) -> "InvitationResponse":
        return InvitationResponse(
            id=inv.id,
            email=inv.email,
            is_admin=inv.isAdmin,
            is_billing_manager=inv.isBillingManager,
            expires_at=inv.expiresAt,
            created_at=inv.createdAt,
            team_ids=inv.teamIds,
        )


class InvitationCreateResponse(InvitationResponse):
    """Extended response returned only from POST /invitations -- includes the token."""

    token: str

    @staticmethod
    def from_db(inv) -> "InvitationCreateResponse":
        return InvitationCreateResponse(
            id=inv.id,
            email=inv.email,
            is_admin=inv.isAdmin,
            is_billing_manager=inv.isBillingManager,
            token=inv.token,
            expires_at=inv.expiresAt,
            created_at=inv.createdAt,
            team_ids=inv.teamIds,
        )
