"""Every REST route either checks the paywall or is listed here as open.

An account without a plan may still sign in, pay, finish onboarding, read and
tidy what it already has, and leave. Anything that runs compute, calls an
outside service or stores new files or rows checks ``enforce_payment_paywall``.
A new route fails ``test_every_route_checks_the_paywall_or_is_listed_open``
until it does one or the other.
"""

import fastapi.testclient
import pytest
from autogpt_libs.auth import requires_admin_user
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute

from backend.api.rest_api import app
from backend.copilot.rate_limit import enforce_payment_paywall

# These check in the handler, because whether the request does paid work
# depends on its input: a chat's model route (the platform needs a plan, a
# ChatGPT connection needs Max via enforce_codex_access_http), or a search
# with a query to embed.
CHECKED_IN_HANDLER = {
    "POST /api/internal/chat/sessions",
    "POST /api/internal/chat/sessions/{session_id}/continue",
    "POST /api/internal/chat/sessions/{session_id}/messages/pending",
    "POST /api/internal/chat/sessions/{session_id}/stream",
    "POST /api/internal/chat/speech",
    "GET /api/internal/integrations/codex/credentials/{credential_id}/account",
    "GET /api/internal/integrations/codex/credentials/{credential_id}/rate-limits",
    "GET /api/internal/search/global",
}

OPEN_WHILE_PAYWALLED = {
    # Signing in and managing the account, its organisations and its links.
    "POST /api/internal/analytics/attribution",
    "GET /api/internal/api-keys",
    "POST /api/internal/api-keys",
    "DELETE /api/internal/api-keys/{key_id}",
    "GET /api/internal/api-keys/{key_id}",
    "PUT /api/internal/api-keys/{key_id}/permissions",
    "POST /api/internal/api-keys/{key_id}/suspend",
    "POST /api/internal/auth/user",
    "POST /api/internal/auth/user/consent",
    "GET /api/internal/auth/user/preferences",
    "POST /api/internal/auth/user/preferences",
    "GET /api/internal/auth/user/timezone",
    "POST /api/internal/auth/user/timezone",
    "PUT /api/internal/chat/transports/default",
    "GET /api/internal/experiments/assignments",
    "GET /api/internal/experiments/assignments/{experiment}",
    "GET /api/internal/invitations/pending",
    "POST /api/internal/invitations/{token}/accept",
    "POST /api/internal/invitations/{token}/decline",
    "DELETE /api/internal/memory",
    "GET /api/internal/memory/experts/{expert_id}/facts",
    "DELETE /api/internal/memory/experts/{expert_id}",
    "DELETE /api/internal/memory/experts/{expert_id}/facts/{fact_uuid}",
    "GET /api/internal/memory/experts/{expert_id}/overview",
    "GET /api/internal/memory/facts",
    "DELETE /api/internal/memory/facts/{fact_uuid}",
    "GET /api/internal/memory/overview",
    "DELETE /api/internal/memory/understanding",
    "GET /api/internal/memory/understanding",
    "GET /api/internal/oauth/app/{client_id}",
    "PATCH /api/internal/oauth/apps/{app_id}/logo",
    "PATCH /api/internal/oauth/apps/{app_id}/status",
    "GET /api/internal/oauth/apps/mine",
    "POST /api/internal/oauth/authorize",
    "GET /api/internal/orgs",
    "POST /api/internal/orgs",
    "DELETE /api/internal/orgs/{org_id}",
    "GET /api/internal/orgs/{org_id}",
    "PATCH /api/internal/orgs/{org_id}",
    "GET /api/internal/orgs/{org_id}/aliases",
    "POST /api/internal/orgs/{org_id}/aliases",
    "POST /api/internal/orgs/{org_id}/convert",
    "GET /api/internal/orgs/{org_id}/invitations",
    "POST /api/internal/orgs/{org_id}/invitations",
    "DELETE /api/internal/orgs/{org_id}/invitations/{invitation_id}",
    "GET /api/internal/orgs/{org_id}/members",
    "DELETE /api/internal/orgs/{org_id}/members/{uid}",
    "PATCH /api/internal/orgs/{org_id}/members/{uid}",
    "POST /api/internal/orgs/{org_id}/transfer-ownership",
    "GET /api/internal/orgs/{org_id}/workspaces",
    "POST /api/internal/orgs/{org_id}/workspaces",
    "DELETE /api/internal/orgs/{org_id}/workspaces/{ws_id}",
    "GET /api/internal/orgs/{org_id}/workspaces/{ws_id}",
    "PATCH /api/internal/orgs/{org_id}/workspaces/{ws_id}",
    "POST /api/internal/orgs/{org_id}/workspaces/{ws_id}/join",
    "POST /api/internal/orgs/{org_id}/workspaces/{ws_id}/leave",
    "GET /api/internal/orgs/{org_id}/workspaces/{ws_id}/members",
    "POST /api/internal/orgs/{org_id}/workspaces/{ws_id}/members",
    "DELETE /api/internal/orgs/{org_id}/workspaces/{ws_id}/members/{uid}",
    "PATCH /api/internal/orgs/{org_id}/workspaces/{ws_id}/members/{uid}",
    "GET /api/internal/platform-linking/links",
    "DELETE /api/internal/platform-linking/links/{link_id}",
    "GET /api/internal/platform-linking/platforms",
    "POST /api/internal/platform-linking/tokens/{token}/confirm",
    "GET /api/internal/platform-linking/tokens/{token}/info",
    "GET /api/internal/platform-linking/user-links",
    "DELETE /api/internal/platform-linking/user-links/{link_id}",
    "POST /api/internal/platform-linking/user-tokens/{token}/confirm",
    "POST /api/internal/push/subscribe",
    "POST /api/internal/push/unsubscribe",
    "GET /api/internal/store/profile",
    "POST /api/internal/store/profile",
    "GET /api/internal/transfers",
    "POST /api/internal/transfers",
    "POST /api/internal/transfers/{transfer_id}/approve",
    "POST /api/internal/transfers/{transfer_id}/execute",
    "POST /api/internal/transfers/{transfer_id}/reject",
    # Paying, and seeing what the account is billed.
    "POST /api/internal/chat/usage/reset",
    "GET /api/internal/credits",
    "PATCH /api/internal/credits",
    "POST /api/internal/credits",
    "GET /api/internal/credits/auto-top-up",
    "POST /api/internal/credits/auto-top-up",
    "GET /api/internal/credits/invoices",
    "GET /api/internal/credits/manage",
    "GET /api/internal/credits/pro-activation/current",
    "POST /api/internal/credits/pro-activation/preview",
    "GET /api/internal/credits/pro-activation/{attempt_id}",
    "POST /api/internal/credits/pro-activation/{attempt_id}/confirm",
    "GET /api/internal/credits/refunds",
    "GET /api/internal/credits/subscription",
    "POST /api/internal/credits/subscription",
    "POST /api/internal/credits/subscription/preview",
    "GET /api/internal/credits/transactions",
    "GET /api/internal/credits/trial",
    "POST /api/internal/credits/trial",
    "POST /api/internal/credits/trial/cancel",
    "POST /api/internal/credits/trial/confirm",
    "POST /api/internal/credits/trial/resume",
    "POST /api/internal/credits/{transaction_key}/refund",
    # Onboarding, which reaches the plan picker only after hiring the team.
    "POST /api/internal/experts",
    "GET /api/internal/onboarding",
    "PATCH /api/internal/onboarding",
    "GET /api/internal/onboarding/agents",
    "DELETE /api/internal/onboarding/brain-dump",
    "POST /api/internal/onboarding/brain-dump/finalize",
    "GET /api/internal/onboarding/brain-dump/intro",
    "POST /api/internal/onboarding/brain-dump/intro/complete",
    "POST /api/internal/onboarding/brain-dump/parts",
    "GET /api/internal/onboarding/brain-dump/recommended-experts",
    "GET /api/internal/onboarding/brain-dump/recommended-providers",
    "GET /api/internal/onboarding/brain-dump/recording",
    "GET /api/internal/onboarding/brain-dump/status",
    "GET /api/internal/onboarding/brain-dump/suggested-tasks",
    "POST /api/internal/onboarding/brain-dump/suggested-tasks",
    "GET /api/internal/onboarding/completed",
    "POST /api/internal/onboarding/profile",
    "POST /api/internal/onboarding/reset",
    "POST /api/internal/onboarding/step",
    # Reading the account's own data, or the catalog.
    "GET /api/internal/blocks",
    "GET /api/internal/briefings/latest",
    "GET /api/internal/builder/blocks",
    "GET /api/internal/builder/blocks/batch",
    "GET /api/internal/builder/categories",
    "GET /api/internal/builder/counts",
    "GET /api/internal/builder/providers",
    "GET /api/internal/builder/suggestions",
    "GET /api/internal/chat/connections",
    "GET /api/internal/chat/model-tiers",
    "GET /api/internal/chat/sessions",
    "GET /api/internal/chat/sessions/{session_id}",
    "GET /api/internal/chat/sessions/{session_id}/at-reset",
    "GET /api/internal/chat/sessions/{session_id}/computer",
    "GET /api/internal/chat/sessions/{session_id}/messages/pending",
    "GET /api/internal/chat/sessions/{session_id}/share/state",
    "GET /api/internal/chat/sessions/{session_id}/stream",
    "GET /api/internal/chat/suggested-prompts",
    "GET /api/internal/chat/transports",
    "GET /api/internal/chat/usage",
    "GET /api/internal/desktop-preview",
    "GET /api/internal/executions",
    "GET /api/internal/executions/cost-summary",
    "GET /api/internal/experts",
    "GET /api/internal/experts/avatars/generations/{job_id}",
    "GET /api/internal/experts/identities",
    "GET /api/internal/experts/pods",
    "GET /api/internal/experts/setup",
    "GET /api/internal/experts/{expert_id}",
    "GET /api/internal/experts/{expert_id}/activity",
    "GET /api/internal/experts/{expert_id}/computer",
    "GET /api/internal/experts/{expert_id}/credentials",
    "GET /api/internal/experts/{expert_id}/detach-preview",
    "GET /api/internal/experts/{expert_id}/onboarding",
    "GET /api/internal/experts/{expert_id}/runs",
    "GET /api/internal/graphs",
    "GET /api/internal/graphs/{graph_id}",
    "GET /api/internal/graphs/{graph_id}/executions",
    "GET /api/internal/graphs/{graph_id}/executions/{graph_exec_id}",
    "GET /api/internal/graphs/{graph_id}/schedules",
    "GET /api/internal/graphs/{graph_id}/versions",
    "GET /api/internal/graphs/{graph_id}/versions/{version}",
    "GET /api/internal/home",
    "GET /api/internal/integrations/codex/device-login/{login_id}",
    "GET /api/internal/integrations/codex/device-login/{login_id}/status",
    "GET /api/internal/integrations/credentials",
    "GET /api/internal/integrations/{provider}/credentials",
    "GET /api/internal/integrations/{provider}/credentials/{cred_id}",
    "POST /api/internal/integrations/{provider}/credentials/{cred_id}/picker-token",
    "GET /api/internal/library/agents",
    "GET /api/internal/library/agents/batch",
    "GET /api/internal/library/agents/by-graph/{graph_id}",
    "GET /api/internal/library/agents/favorites",
    "GET /api/internal/library/agents/marketplace/{store_listing_version_id}",
    "GET /api/internal/library/agents/{library_agent_id}",
    "GET /api/internal/library/agents/{library_agent_id}/triggers",
    "GET /api/internal/library/folders",
    "GET /api/internal/library/folders/tree",
    "GET /api/internal/library/folders/{folder_id}",
    "GET /api/internal/library/presets",
    "GET /api/internal/library/presets/{preset_id}",
    "GET /api/internal/review/execution/{graph_exec_id}",
    "GET /api/internal/review/pending",
    "GET /api/internal/review/session/{chat_session_id}",
    "GET /api/internal/review/session/{chat_session_id}/answered",
    "GET /api/internal/schedules",
    "GET /api/internal/schedules/followups",
    "GET /api/internal/schedules/followups/paused",
    "GET /api/internal/schedules/followups/{schedule_key}/cadence",
    "GET /api/internal/skill-learning/decisions",
    "GET /api/internal/skill-learning/history",
    "GET /api/internal/skill-learning/skills/{name}",
    "GET /api/internal/skill-learning/status",
    "GET /api/internal/skills",
    "GET /api/internal/skills/{name}",
    "GET /api/internal/skills/{name}/package",
    "GET /api/internal/store/listings/versions/{store_listing_version_id}/graph",
    "GET /api/internal/store/my-unpublished-agents",
    "GET /api/internal/store/skills/submissions",
    "GET /api/internal/store/submissions",
    "GET /api/internal/store/submissions/media/{owner_user_id}/{media_type}/{filename}",
    "GET /api/internal/workspace/files",
    "GET /api/internal/workspace/files/{file_id}/download",
    "GET /api/internal/workspace/folders",
    "GET /api/internal/workspace/storage/usage",
    # Stopping, deleting or revoking.
    "DELETE /api/internal/chat/sessions/{session_id}",
    "DELETE /api/internal/chat/sessions/{session_id}/at-reset/{kind}",
    "POST /api/internal/chat/sessions/{session_id}/cancel",
    "DELETE /api/internal/chat/sessions/{session_id}/share",
    "DELETE /api/internal/chat/sessions/{session_id}/stream",
    "DELETE /api/internal/executions/{graph_exec_id}",
    "DELETE /api/internal/experts/{expert_id}",
    "DELETE /api/internal/experts/{expert_id}/credentials/{credential_id}",
    "DELETE /api/internal/experts/{expert_id}/workflows/{workflow_id}",
    "DELETE /api/internal/graphs/{graph_id}",
    "DELETE /api/internal/graphs/{graph_id}/executions/{graph_exec_id}/share",
    "POST /api/internal/graphs/{graph_id}/executions/{graph_exec_id}/stop",
    "POST /api/internal/integrations/codex/device-login/{login_id}/cancel",
    "DELETE /api/internal/integrations/{provider}/credentials/{cred_id}",
    "DELETE /api/internal/library/agents/{library_agent_id}",
    "DELETE /api/internal/library/folders/{folder_id}",
    "DELETE /api/internal/library/presets/{preset_id}",
    "DELETE /api/internal/schedules/{schedule_id}",
    "POST /api/internal/skill-learning/sources/{source_kind}/{source_ref}/exclude",
    "DELETE /api/internal/skills/{name}",
    "DELETE /api/internal/store/submissions/{submission_id}",
    "DELETE /api/internal/workspace/files/{file_id}",
    "DELETE /api/internal/workspace/folders/{folder_id}",
    # Organising what the account already has: names, places, settings.
    "PATCH /api/internal/chat/sessions/{session_id}/assign-user",
    "PUT /api/internal/chat/sessions/{session_id}/connection",
    "PUT /api/internal/chat/sessions/{session_id}/credential-selection",
    "POST /api/internal/chat/sessions/{session_id}/feedback",
    "PATCH /api/internal/chat/sessions/{session_id}/pinned",
    "POST /api/internal/chat/sessions/{session_id}/share",
    "PATCH /api/internal/chat/sessions/{session_id}/title",
    "POST /api/internal/experts/pods",
    "PATCH /api/internal/experts/{expert_id}/avatar",
    "PATCH /api/internal/experts/{expert_id}/budget",
    "PATCH /api/internal/experts/{expert_id}/mode",
    "PATCH /api/internal/experts/{expert_id}/pod",
    "PATCH /api/internal/experts/{expert_id}/soul",
    "POST /api/internal/graphs/{graph_id}/executions/{graph_exec_id}/share",
    "PATCH /api/internal/graphs/{graph_id}/settings",
    "PATCH /api/internal/library/agents/{library_agent_id}",
    "POST /api/internal/library/folders",
    "POST /api/internal/library/folders/agents/bulk-move",
    "PATCH /api/internal/library/folders/{folder_id}",
    "POST /api/internal/library/folders/{folder_id}/move",
    "POST /api/internal/skill-learning/experts/{expert_id}/policy",
    "POST /api/internal/skill-learning/skills/{name}/outcome",
    "POST /api/internal/skill-learning/skills/{name}/policy",
    "PATCH /api/internal/workspace/files/{file_id}",
    "POST /api/internal/workspace/folders",
    "POST /api/internal/workspace/folders/files/bulk-move",
    "PATCH /api/internal/workspace/folders/{folder_id}",
    # No signed-in user to check: public pages, provider webhooks and token
    # endpoints. A webhook's runs check their owner (add_graph_execution).
    "GET /.well-known/oauth-protected-resource/api/v2/mcp",
    "POST /api/internal/auth/email/send",
    "POST /api/internal/auth/email/suppression",
    "POST /api/internal/auth/user/preferences/from-email",
    "GET /api/internal/chat/config/ttl",
    "GET /api/internal/chat/schema/tool-responses",
    "POST /api/internal/credits/stripe_webhook",
    "POST /api/internal/email/",
    "POST /api/internal/email/mailerlite/webhook",
    "POST /api/internal/email/unsubscribe",
    "GET /api/internal/experts/templates",
    "GET /api/internal/feature-flags",
    "GET /api/internal/integrations/providers",
    "GET /api/internal/integrations/providers/constants",
    "GET /api/internal/integrations/providers/enum-example",
    "GET /api/internal/integrations/providers/names",
    "GET /api/internal/integrations/providers/system",
    "POST /api/internal/integrations/{provider}/webhooks/{webhook_id}/ingress",
    "POST /api/internal/oauth/introspect",
    "POST /api/internal/oauth/revoke",
    "POST /api/internal/oauth/token",
    "GET /api/internal/public/shared/chats/{share_token}",
    "GET /api/internal/public/shared/chats/{share_token}/files/{file_id}/download",
    "GET /api/internal/public/shared/chats/{share_token}/messages",
    "GET /api/internal/public/shared/{share_token}",
    "GET /api/internal/public/shared/{share_token}/files/{file_id}/download",
    "GET /api/internal/push/vapid-key",
    "GET /api/internal/store/agents",
    "GET /api/internal/store/agents/{username}/{agent_name}",
    "GET /api/internal/store/categories",
    "GET /api/internal/store/creators",
    "GET /api/internal/store/creators/{username}",
    "GET /api/internal/store/listings/versions/{store_listing_version_id}",
    "GET /api/internal/store/listings/versions/{store_listing_version_id}/graph/download",
    "GET /api/internal/store/media/{user_id}/{media_type}/{filename}",
    "GET /api/internal/store/metrics/cache",
    "GET /api/internal/store/search",
    "GET /api/internal/store/skills",
    "GET /api/internal/store/skills/{slug}",
    "GET /api/internal/store/skills/{slug}/files/{path:path}",
    "GET /health",
}

# One closed route per router, as the client calls it.
CLOSED_SAMPLES = [
    "GET /api/internal/builder/search",
    "POST /api/internal/chat/sessions/s-1/at-reset",
    "POST /api/internal/chat/sessions/s-1/desktop",
    "POST /api/internal/experts/avatars/generations",
    "POST /api/internal/experts/raise",
    "POST /api/internal/files/upload",
    "POST /api/internal/graphs",
    "POST /api/internal/graphs/g-1/schedules",
    "GET /api/internal/integrations/github/login",
    "POST /api/internal/library/agents/la-1/fork",
    "POST /api/internal/library/presets/setup-trigger",
    "POST /api/internal/mcp/discover-tools",
    "POST /api/internal/oauth/apps/a-1/logo/upload",
    "POST /api/internal/otto/ask",
    "POST /api/internal/review/action",
    "GET /api/internal/search/global?q=weekly report",
    "POST /api/internal/skill-learning/run",
    "POST /api/internal/skills",
    "POST /api/internal/store/skills/submissions",
    "POST /api/internal/store/submissions/media",
    "POST /api/internal/workspace/files/upload",
    "GET /api/internal/workspace/files/f-1/preview",
]


def test_every_route_checks_the_paywall_or_is_listed_open():
    unlisted, checked_but_listed = [], []
    for key, route in _routes_with_a_user():
        checked = _depends_on(route.dependant, enforce_payment_paywall)
        listed = key in OPEN_WHILE_PAYWALLED or key in CHECKED_IN_HANDLER
        if checked and listed:
            checked_but_listed.append(key)
        elif not checked and not listed:
            unlisted.append(key)

    assert not unlisted, (
        "Add Depends(enforce_payment_paywall) to these routes, or list them in "
        f"OPEN_WHILE_PAYWALLED with the reason they stay open: {sorted(unlisted)}"
    )
    assert not checked_but_listed, (
        f"These routes check the paywall, so drop them from the open lists: "
        f"{sorted(checked_but_listed)}"
    )


def test_the_open_lists_name_only_routes_that_exist():
    served = {key for key, _ in _routes_with_a_user()}
    assert not (OPEN_WHILE_PAYWALLED | CHECKED_IN_HANDLER) - served
    assert OPEN_WHILE_PAYWALLED.isdisjoint(CHECKED_IN_HANDLER)


@pytest.mark.parametrize("call", CLOSED_SAMPLES)
def test_an_account_without_a_plan_gets_402_from_a_closed_route(
    call: str, mock_jwt_user, paywalled_user, mocker
):
    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]
    try:
        method, path = call.split(" ", 1)
        response = fastapi.testclient.TestClient(app).request(method, path)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 402, response.text
    assert "subscription is required" in response.json()["detail"]


def _routes_with_a_user():
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if _depends_on(route.dependant, requires_admin_user):
            continue
        for method in route.methods:
            yield f"{method} {route.path}", route


def _depends_on(dependant: Dependant, call) -> bool:
    return any(
        dep.call is call or _depends_on(dep, call) for dep in dependant.dependencies
    )
