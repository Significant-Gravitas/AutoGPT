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
    "POST /api/chat/sessions",
    "POST /api/chat/sessions/{session_id}/continue",
    "POST /api/chat/sessions/{session_id}/messages/pending",
    "POST /api/chat/sessions/{session_id}/stream",
    "POST /api/chat/speech",
    "GET /api/integrations/codex/credentials/{credential_id}/account",
    "GET /api/integrations/codex/credentials/{credential_id}/rate-limits",
    "GET /api/search/global",
}

OPEN_WHILE_PAYWALLED = {
    # Signing in and managing the account, its organisations and its links.
    "POST /api/analytics/attribution",
    "GET /api/api-keys",
    "POST /api/api-keys",
    "DELETE /api/api-keys/{key_id}",
    "GET /api/api-keys/{key_id}",
    "PUT /api/api-keys/{key_id}/permissions",
    "POST /api/api-keys/{key_id}/suspend",
    "POST /api/auth/user",
    "POST /api/auth/user/consent",
    "GET /api/auth/user/preferences",
    "POST /api/auth/user/preferences",
    "GET /api/auth/user/timezone",
    "POST /api/auth/user/timezone",
    "PUT /api/chat/transports/default",
    "GET /api/experiments/assignments",
    "POST /api/experiments/assignments",
    "GET /api/invitations/pending",
    "POST /api/invitations/{token}/accept",
    "POST /api/invitations/{token}/decline",
    "DELETE /api/memory",
    "GET /api/memory/experts/{expert_id}/facts",
    "DELETE /api/memory/experts/{expert_id}",
    "DELETE /api/memory/experts/{expert_id}/facts/{fact_uuid}",
    "GET /api/memory/experts/{expert_id}/overview",
    "GET /api/memory/facts",
    "DELETE /api/memory/facts/{fact_uuid}",
    "GET /api/memory/overview",
    "GET /api/oauth/app/{client_id}",
    "PATCH /api/oauth/apps/{app_id}/logo",
    "PATCH /api/oauth/apps/{app_id}/status",
    "GET /api/oauth/apps/mine",
    "POST /api/oauth/authorize",
    "GET /api/orgs",
    "POST /api/orgs",
    "DELETE /api/orgs/{org_id}",
    "GET /api/orgs/{org_id}",
    "PATCH /api/orgs/{org_id}",
    "GET /api/orgs/{org_id}/aliases",
    "POST /api/orgs/{org_id}/aliases",
    "POST /api/orgs/{org_id}/convert",
    "GET /api/orgs/{org_id}/invitations",
    "POST /api/orgs/{org_id}/invitations",
    "DELETE /api/orgs/{org_id}/invitations/{invitation_id}",
    "GET /api/orgs/{org_id}/members",
    "POST /api/orgs/{org_id}/members",
    "DELETE /api/orgs/{org_id}/members/{uid}",
    "PATCH /api/orgs/{org_id}/members/{uid}",
    "POST /api/orgs/{org_id}/transfer-ownership",
    "GET /api/orgs/{org_id}/workspaces",
    "POST /api/orgs/{org_id}/workspaces",
    "DELETE /api/orgs/{org_id}/workspaces/{ws_id}",
    "GET /api/orgs/{org_id}/workspaces/{ws_id}",
    "PATCH /api/orgs/{org_id}/workspaces/{ws_id}",
    "POST /api/orgs/{org_id}/workspaces/{ws_id}/join",
    "POST /api/orgs/{org_id}/workspaces/{ws_id}/leave",
    "GET /api/orgs/{org_id}/workspaces/{ws_id}/members",
    "POST /api/orgs/{org_id}/workspaces/{ws_id}/members",
    "DELETE /api/orgs/{org_id}/workspaces/{ws_id}/members/{uid}",
    "PATCH /api/orgs/{org_id}/workspaces/{ws_id}/members/{uid}",
    "GET /api/platform-linking/links",
    "DELETE /api/platform-linking/links/{link_id}",
    "GET /api/platform-linking/platforms",
    "POST /api/platform-linking/tokens/{token}/confirm",
    "GET /api/platform-linking/tokens/{token}/info",
    "GET /api/platform-linking/user-links",
    "DELETE /api/platform-linking/user-links/{link_id}",
    "POST /api/platform-linking/user-tokens/{token}/confirm",
    "POST /api/push/subscribe",
    "POST /api/push/unsubscribe",
    "GET /api/store/profile",
    "POST /api/store/profile",
    "GET /api/transfers",
    "POST /api/transfers",
    "POST /api/transfers/{transfer_id}/approve",
    "POST /api/transfers/{transfer_id}/execute",
    "POST /api/transfers/{transfer_id}/reject",
    # Paying, and seeing what the account is billed.
    "POST /api/chat/usage/reset",
    "GET /api/credits",
    "PATCH /api/credits",
    "POST /api/credits",
    "GET /api/credits/auto-top-up",
    "POST /api/credits/auto-top-up",
    "GET /api/credits/invoices",
    "GET /api/credits/manage",
    "GET /api/credits/pro-activation/current",
    "POST /api/credits/pro-activation/preview",
    "GET /api/credits/pro-activation/{attempt_id}",
    "POST /api/credits/pro-activation/{attempt_id}/confirm",
    "GET /api/credits/refunds",
    "GET /api/credits/subscription",
    "POST /api/credits/subscription",
    "POST /api/credits/subscription/preview",
    "GET /api/credits/transactions",
    "GET /api/credits/trial",
    "POST /api/credits/trial",
    "POST /api/credits/trial/cancel",
    "POST /api/credits/trial/confirm",
    "POST /api/credits/trial/resume",
    "POST /api/credits/{transaction_key}/refund",
    # Onboarding, which reaches the plan picker only after hiring the team.
    "POST /api/experts",
    "GET /api/onboarding",
    "PATCH /api/onboarding",
    "GET /api/onboarding/agents",
    "DELETE /api/onboarding/brain-dump",
    "POST /api/onboarding/brain-dump/finalize",
    "GET /api/onboarding/brain-dump/intro",
    "POST /api/onboarding/brain-dump/intro/complete",
    "POST /api/onboarding/brain-dump/parts",
    "GET /api/onboarding/brain-dump/recommended-experts",
    "GET /api/onboarding/brain-dump/recommended-providers",
    "GET /api/onboarding/brain-dump/recording",
    "GET /api/onboarding/brain-dump/status",
    "GET /api/onboarding/brain-dump/suggested-tasks",
    "POST /api/onboarding/brain-dump/suggested-tasks",
    "GET /api/onboarding/completed",
    "POST /api/onboarding/profile",
    "POST /api/onboarding/reset",
    "POST /api/onboarding/step",
    # Reading the account's own data, or the catalog.
    "GET /api/blocks",
    "GET /api/briefings/latest",
    "GET /api/builder/blocks",
    "GET /api/builder/blocks/batch",
    "GET /api/builder/categories",
    "GET /api/builder/counts",
    "GET /api/builder/providers",
    "GET /api/builder/suggestions",
    "GET /api/chat/connections",
    "GET /api/chat/model-tiers",
    "GET /api/chat/sessions",
    "GET /api/chat/sessions/{session_id}",
    "GET /api/chat/sessions/{session_id}/at-reset",
    "GET /api/chat/sessions/{session_id}/computer",
    "GET /api/chat/sessions/{session_id}/messages/pending",
    "GET /api/chat/sessions/{session_id}/share/state",
    "GET /api/chat/sessions/{session_id}/stream",
    "GET /api/chat/suggested-prompts",
    "GET /api/chat/transports",
    "GET /api/chat/usage",
    "GET /api/desktop-preview",
    "GET /api/executions",
    "GET /api/executions/cost-summary",
    "GET /api/experts",
    "GET /api/experts/avatars/generations/{job_id}",
    "GET /api/experts/identities",
    "GET /api/experts/pods",
    "GET /api/experts/setup",
    "GET /api/experts/{expert_id}",
    "GET /api/experts/{expert_id}/activity",
    "GET /api/experts/{expert_id}/computer",
    "GET /api/experts/{expert_id}/credentials",
    "GET /api/experts/{expert_id}/detach-preview",
    "GET /api/experts/{expert_id}/onboarding",
    "GET /api/experts/{expert_id}/runs",
    "GET /api/graphs",
    "GET /api/graphs/{graph_id}",
    "GET /api/graphs/{graph_id}/executions",
    "GET /api/graphs/{graph_id}/executions/{graph_exec_id}",
    "GET /api/graphs/{graph_id}/schedules",
    "GET /api/graphs/{graph_id}/versions",
    "GET /api/graphs/{graph_id}/versions/{version}",
    "GET /api/home",
    "GET /api/integrations/codex/device-login/{login_id}",
    "GET /api/integrations/codex/device-login/{login_id}/status",
    "GET /api/integrations/credentials",
    "GET /api/integrations/{provider}/credentials",
    "GET /api/integrations/{provider}/credentials/{cred_id}",
    "POST /api/integrations/{provider}/credentials/{cred_id}/picker-token",
    "GET /api/library/agents",
    "GET /api/library/agents/batch",
    "GET /api/library/agents/by-graph/{graph_id}",
    "GET /api/library/agents/favorites",
    "GET /api/library/agents/marketplace/{store_listing_version_id}",
    "GET /api/library/agents/{library_agent_id}",
    "GET /api/library/agents/{library_agent_id}/triggers",
    "GET /api/library/folders",
    "GET /api/library/folders/tree",
    "GET /api/library/folders/{folder_id}",
    "GET /api/library/presets",
    "GET /api/library/presets/{preset_id}",
    "GET /api/review/execution/{graph_exec_id}",
    "GET /api/review/pending",
    "GET /api/review/session/{chat_session_id}",
    "GET /api/review/session/{chat_session_id}/answered",
    "GET /api/schedules",
    "GET /api/schedules/followups",
    "GET /api/schedules/followups/paused",
    "GET /api/schedules/followups/{schedule_key}/cadence",
    "GET /api/skill-learning/decisions",
    "GET /api/skill-learning/history",
    "GET /api/skill-learning/skills/{name}",
    "GET /api/skill-learning/status",
    "GET /api/skills",
    "GET /api/skills/{name}",
    "GET /api/skills/{name}/package",
    "GET /api/store/listings/versions/{store_listing_version_id}/graph",
    "GET /api/store/my-unpublished-agents",
    "GET /api/store/skills/submissions",
    "GET /api/store/submissions",
    "GET /api/store/submissions/media/{owner_user_id}/{media_type}/{filename}",
    "GET /api/workspace/files",
    "GET /api/workspace/files/{file_id}/download",
    "GET /api/workspace/folders",
    "GET /api/workspace/storage/usage",
    # Stopping, deleting or revoking.
    "DELETE /api/chat/sessions/{session_id}",
    "DELETE /api/chat/sessions/{session_id}/at-reset/{kind}",
    "POST /api/chat/sessions/{session_id}/cancel",
    "DELETE /api/chat/sessions/{session_id}/share",
    "DELETE /api/chat/sessions/{session_id}/stream",
    "DELETE /api/executions/{graph_exec_id}",
    "DELETE /api/experts/{expert_id}",
    "DELETE /api/experts/{expert_id}/credentials/{credential_id}",
    "DELETE /api/experts/{expert_id}/workflows/{workflow_id}",
    "DELETE /api/graphs/{graph_id}",
    "DELETE /api/graphs/{graph_id}/executions/{graph_exec_id}/share",
    "POST /api/graphs/{graph_id}/executions/{graph_exec_id}/stop",
    "POST /api/integrations/codex/device-login/{login_id}/cancel",
    "DELETE /api/integrations/{provider}/credentials/{cred_id}",
    "DELETE /api/library/agents/{library_agent_id}",
    "DELETE /api/library/folders/{folder_id}",
    "DELETE /api/library/presets/{preset_id}",
    "DELETE /api/schedules/{schedule_id}",
    "POST /api/skill-learning/sources/{source_kind}/{source_ref}/exclude",
    "DELETE /api/skills/{name}",
    "DELETE /api/store/submissions/{submission_id}",
    "DELETE /api/workspace/files/{file_id}",
    "DELETE /api/workspace/folders/{folder_id}",
    # Organising what the account already has: names, places, settings.
    "PATCH /api/chat/sessions/{session_id}/assign-user",
    "PUT /api/chat/sessions/{session_id}/connection",
    "PUT /api/chat/sessions/{session_id}/credential-selection",
    "POST /api/chat/sessions/{session_id}/feedback",
    "PATCH /api/chat/sessions/{session_id}/pinned",
    "POST /api/chat/sessions/{session_id}/share",
    "PATCH /api/chat/sessions/{session_id}/title",
    "POST /api/experts/pods",
    "PATCH /api/experts/{expert_id}/avatar",
    "PATCH /api/experts/{expert_id}/budget",
    "PATCH /api/experts/{expert_id}/mode",
    "PATCH /api/experts/{expert_id}/pod",
    "PATCH /api/experts/{expert_id}/soul",
    "POST /api/graphs/{graph_id}/executions/{graph_exec_id}/share",
    "PATCH /api/graphs/{graph_id}/settings",
    "PATCH /api/library/agents/{library_agent_id}",
    "POST /api/library/folders",
    "POST /api/library/folders/agents/bulk-move",
    "PATCH /api/library/folders/{folder_id}",
    "POST /api/library/folders/{folder_id}/move",
    "POST /api/skill-learning/experts/{expert_id}/policy",
    "POST /api/skill-learning/skills/{name}/outcome",
    "POST /api/skill-learning/skills/{name}/policy",
    "PATCH /api/workspace/files/{file_id}",
    "POST /api/workspace/folders",
    "POST /api/workspace/folders/files/bulk-move",
    "PATCH /api/workspace/folders/{folder_id}",
    # No signed-in user to check: public pages, provider webhooks and token
    # endpoints. A webhook's runs check their owner (add_graph_execution).
    "POST /api/auth/email/send",
    "POST /api/auth/user/preferences/from-email",
    "GET /api/chat/config/ttl",
    "GET /api/chat/schema/tool-responses",
    "POST /api/credits/stripe_webhook",
    "POST /api/email/",
    "POST /api/email/mailerlite/webhook",
    "POST /api/email/unsubscribe",
    "GET /api/experts/templates",
    "GET /api/feature-flags",
    "GET /api/integrations/providers",
    "GET /api/integrations/providers/constants",
    "GET /api/integrations/providers/enum-example",
    "GET /api/integrations/providers/names",
    "GET /api/integrations/providers/system",
    "POST /api/integrations/{provider}/webhooks/{webhook_id}/ingress",
    "POST /api/oauth/introspect",
    "POST /api/oauth/revoke",
    "POST /api/oauth/token",
    "GET /api/public/shared/chats/{share_token}",
    "GET /api/public/shared/chats/{share_token}/files/{file_id}/download",
    "GET /api/public/shared/chats/{share_token}/messages",
    "GET /api/public/shared/{share_token}",
    "GET /api/public/shared/{share_token}/files/{file_id}/download",
    "GET /api/push/vapid-key",
    "GET /api/store/agents",
    "GET /api/store/agents/{username}/{agent_name}",
    "GET /api/store/categories",
    "GET /api/store/creators",
    "GET /api/store/creators/{username}",
    "GET /api/store/listings/versions/{store_listing_version_id}",
    "GET /api/store/listings/versions/{store_listing_version_id}/graph/download",
    "GET /api/store/media/{user_id}/{media_type}/{filename}",
    "GET /api/store/metrics/cache",
    "GET /api/store/search",
    "GET /api/store/skills",
    "GET /api/store/skills/{slug}",
    "GET /api/store/skills/{slug}/files/{path:path}",
    "GET /health",
}

# One closed route per router, as the client calls it.
CLOSED_SAMPLES = [
    "GET /api/builder/search",
    "POST /api/chat/sessions/s-1/at-reset",
    "POST /api/chat/sessions/s-1/desktop",
    "POST /api/experts/avatars/generations",
    "POST /api/experts/raise",
    "POST /api/files/upload",
    "POST /api/graphs",
    "POST /api/graphs/g-1/schedules",
    "GET /api/integrations/github/login",
    "POST /api/library/agents/la-1/fork",
    "POST /api/library/presets/setup-trigger",
    "POST /api/mcp/discover-tools",
    "POST /api/oauth/apps/a-1/logo/upload",
    "POST /api/otto/ask",
    "POST /api/review/action",
    "GET /api/search/global?q=weekly report",
    "POST /api/skill-learning/run",
    "POST /api/skills",
    "POST /api/store/skills/submissions",
    "POST /api/store/submissions/media",
    "POST /api/workspace/files/upload",
    "GET /api/workspace/files/f-1/preview",
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
