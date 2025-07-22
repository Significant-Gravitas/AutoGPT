"""
Shared configuration for all Airtable blocks using the SDK pattern.
"""

import os
from enum import Enum

from backend.sdk import BlockCostType, ProviderBuilder

from ._oauth import AirtableOAuthHandler
from ._webhook import AirtableWebhookManager


class AirtableScope(str, Enum):
    # Basic scopes
    DATA_RECORDS_READ = "data.records:read"
    DATA_RECORDS_WRITE = "data.records:write"
    DATA_RECORD_COMMENTS_READ = "data.recordComments:read"
    DATA_RECORD_COMMENTS_WRITE = "data.recordComments:write"
    SCHEMA_BASES_READ = "schema.bases:read"
    SCHEMA_BASES_WRITE = "schema.bases:write"
    WEBHOOK_MANAGE = "webhook:manage"
    BLOCK_MANAGE = "block:manage"
    USER_EMAIL_READ = "user.email:read"

    # Enterprise member scopes
    ENTERPRISE_GROUPS_READ = "enterprise.groups:read"
    WORKSPACES_AND_BASES_READ = "workspacesAndBases:read"
    WORKSPACES_AND_BASES_WRITE = "workspacesAndBases:write"
    WORKSPACES_AND_BASES_SHARES_MANAGE = "workspacesAndBases.shares:manage"

    # Enterprise admin scopes
    ENTERPRISE_SCIM_USERS_AND_GROUPS_MANAGE = "enterprise.scim.usersAndGroups:manage"
    ENTERPRISE_AUDIT_LOGS_READ = "enterprise.auditLogs:read"
    ENTERPRISE_CHANGE_EVENTS_READ = "enterprise.changeEvents:read"
    ENTERPRISE_EXPORTS_MANAGE = "enterprise.exports:manage"
    ENTERPRISE_ACCOUNT_READ = "enterprise.account:read"
    ENTERPRISE_ACCOUNT_WRITE = "enterprise.account:write"
    ENTERPRISE_USER_READ = "enterprise.user:read"
    ENTERPRISE_USER_WRITE = "enterprise.user:write"
    ENTERPRISE_GROUPS_MANAGE = "enterprise.groups:manage"
    WORKSPACES_AND_BASES_MANAGE = "workspacesAndBases:manage"
    HYPERDB_RECORDS_READ = "hyperDB.records:read"
    HYPERDB_RECORDS_WRITE = "hyperDB.records:write"


# Configure the Airtable provider with API key authentication
builder = (
    ProviderBuilder("airtable")
    .with_api_key("AIRTABLE_API_KEY", "Airtable Personal Access Token")
    .with_webhook_manager(AirtableWebhookManager)
    .with_base_cost(1, BlockCostType.RUN)
)


# Check if Linear OAuth is configured
client_id = os.getenv("AIRTABLE_CLIENT_ID")
client_secret = os.getenv("AIRTABLE_CLIENT_SECRET")
AIRTABLE_OAUTH_IS_CONFIGURED = bool(client_id and client_secret)

# Linear only supports OAuth authentication
if AIRTABLE_OAUTH_IS_CONFIGURED:
    builder = builder.with_oauth(
        AirtableOAuthHandler,
        scopes=[
            AirtableScope.DATA_RECORDS_READ,
            AirtableScope.DATA_RECORDS_WRITE,
            AirtableScope.SCHEMA_BASES_READ,
            AirtableScope.SCHEMA_BASES_WRITE,
            AirtableScope.WEBHOOK_MANAGE,
        ],
        client_id_env_var="AIRTABLE_CLIENT_ID",
        client_secret_env_var="AIRTABLE_CLIENT_SECRET",
    )

# Build the provider
airtable = builder.build()
