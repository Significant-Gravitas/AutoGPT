"use client";

import { useGetV1ListCredentials } from "@/app/api/__generated__/endpoints/integrations/integrations";
import {
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
  postV2StoreABearerTokenForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import type { SetupRequirementsResponse } from "@/app/api/__generated__/models/setupRequirementsResponse";
import { Button } from "@/components/atoms/Button/Button";
import { openOAuthPopup } from "@/lib/oauth-popup";
import { CredentialsProvidersContext } from "@/providers/agent-credentials/credentials-provider";
import { useContext, useEffect, useRef, useState } from "react";
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { ContentMessage } from "../../../../components/ToolAccordion/AccordionContent";

function normalizeMcpUrl(url: string): string {
  // Mirrors backend ``normalize_mcp_url`` (helpers.py) so a stored cred
  // for ``https://mcp.sentry.dev/mcp`` matches a card emitted with the
  // same URL whether or not the trailing slash is present.
  return url.trim().replace(/\/+$/, "");
}
interface Props {
  output: SetupRequirementsResponse;
  /**
   * Message sent to the chat after a successful connection.
   * Defaults to a generic "credentials connected, please retry" message.
   */
  retryInstruction?: string;
}

/**
 * Credential setup card for MCP servers — replaces the generic
 * SetupRequirementsCard (which uses the standard integration OAuth route)
 * with the MCP-specific OAuth flow (`/api/v2/mcp/oauth/*`).
 *
 * OAuth flow: initiate login → popup → exchange code for tokens.
 * Fallback: if the server doesn't support MCP OAuth (400), prompts the user
 * to provide an API token manually.
 */
export function MCPSetupCard({ output, retryInstruction }: Props) {
  const { onSend } = useCopilotChatActions();
  const allProviders = useContext(CredentialsProvidersContext);

  // setup_info.agent_id is set to the server_url in the backend
  const serverUrl = output.setup_info.agent_id;
  // agent_name is computed by the backend as the display name for the service
  const service = output.setup_info.agent_name;

  // Initial connection state comes from the backend.  When the model
  // calls `run_mcp_tool` with `surface_connect_card=true`, the response's
  // `has_all_credentials` reflects whether the user already has a stored
  // credential for this server — so we render "Connected — Reconnect"
  // straight away instead of a bare Connect button.
  const initiallyConnected = Boolean(
    output.setup_info?.user_readiness?.has_all_credentials,
  );

  // The persisted backend state above is a snapshot from the moment the
  // card was emitted.  On chat refresh that snapshot is stale — the user
  // may have completed sign-in in a prior session.  Re-fetch the live
  // cred list and OR it into ``connected`` so a previously-connected
  // server still renders the "Connected — Reconnect" pill after a
  // page reload.  We OR rather than override: once the user clicks
  // Connect successfully in this component, ``localConnected`` stays
  // true even if the live-cred query is briefly stale.
  const normalizedServer = normalizeMcpUrl(serverUrl);
  const { data: liveCredsRes } = useGetV1ListCredentials({
    query: {
      select: (res) => (res.status === 200 ? res.data : null),
      // No staleTime — when this card mounts (e.g. immediately after the
      // backend's ``invalidate_mcp_credential`` deleted a stale row),
      // we want a fresh fetch.  A non-zero staleTime would let the
      // previously-cached "cred exists" override the post-invalidation
      // truth, briefly rendering "Connected" on a card whose
      // ``has_all_credentials=false`` snapshot is the authoritative
      // post-invalidation state.
      refetchOnMount: "always",
    },
  });
  // Tri-state: ``true``/``false`` when the live API responded, ``"unknown"``
  // while loading or after a network/auth failure (``select`` returned
  // ``null``).  Treating an unknown live state as ``false`` would override
  // a still-valid persisted snapshot — see review for the
  // initiallyConnected=false + 5xx race that surfaces a bare Connect
  // button despite an existing cred.
  const liveHasCred: boolean | "unknown" = !Array.isArray(liveCredsRes)
    ? "unknown"
    : liveCredsRes.some(
        (c) =>
          c.provider === "mcp" &&
          typeof c.host === "string" &&
          normalizeMcpUrl(c.host) === normalizedServer,
      );

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showManualToken, setShowManualToken] = useState(false);
  const [manualToken, setManualToken] = useState("");
  // ``localConnected`` is set ONLY when the user successfully completes
  // OAuth / manual-token in this component instance.  It is NOT seeded
  // from ``initiallyConnected`` — that path is handled via ``liveSays``
  // below.  Seeding from the persisted snapshot would shadow a later
  // ``liveHasCred=false`` (e.g. cred revoked) and keep the Connected
  // pill stuck.
  const [localConnected, setLocalConnected] = useState(false);
  // When Reconnect fails (or any in-card flow errors out), force the
  // not-connected branch even though the live cred row still exists —
  // otherwise ``liveHasCred=true`` would keep the pill rendered and the
  // user couldn't see the error banner or the manual-token input.  Reset
  // on the next attempt so the user can retry.
  const [forceDisconnected, setForceDisconnected] = useState(false);
  const oauthAbortRef = useRef<(() => void) | null>(null);

  // Combined view:
  //   1. ``forceDisconnected`` (set by the catch block) wins.
  //   2. ``localConnected`` (just completed sign-in in this component) wins.
  //   3. Live API truth wins over the persisted snapshot — a true tells
  //      us the cred is there, a false tells us it isn't.
  //   4. When live state is unknown (loading/network/auth error), fall
  //      back to the persisted ``initiallyConnected`` snapshot rather
  //      than defaulting to disconnected.
  const liveSays = liveHasCred === "unknown" ? initiallyConnected : liveHasCred;
  const connected = !forceDisconnected && (localConnected || liveSays);
  // Setter compatible with the existing call-sites — they only ever set
  // ``true`` after a successful flow or ``false`` to drop the pill.
  const setConnected = setLocalConnected;

  // Abort any in-progress OAuth popup when the component unmounts.
  useEffect(() => () => oauthAbortRef.current?.(), []);

  async function handleConnect() {
    // Re-entrancy guard: a rapid double-click would otherwise race the
    // two in-flight ``handleConnect`` invocations — the second one aborts
    // the first's popup (``oauthAbortRef.current?.()``) but the first's
    // ``await promise`` then rejects with ``OAUTH_ERROR_FLOW_CANCELED``,
    // which flips ``forceDisconnected=true`` even though the second
    // attempt is still alive.  Bail out cheaply when a flow is already
    // running.  Button is also ``disabled={loading}`` but disabled
    // <button> elements still fire ``click`` in some browsers.
    if (loading) return;
    setError(null);
    // Reset showManualToken so a prior 400 doesn't keep the input visible
    // when a later attempt fails with a non-400 (e.g. network) error.
    setShowManualToken(false);
    setLoading(true);
    oauthAbortRef.current?.();

    try {
      const loginRes = await postV2InitiateOauthLoginForAnMcpServer({
        server_url: serverUrl,
      });
      if (!(loginRes.status >= 200 && loginRes.status < 300)) {
        const d =
          loginRes.data && typeof loginRes.data === "object"
            ? loginRes.data
            : {};
        throw { status: loginRes.status, ...d };
      }
      const { login_url, state_token } = loginRes.data as {
        login_url: string;
        state_token: string;
      };

      const { promise, cleanup } = openOAuthPopup(login_url, {
        stateToken: state_token,
        useCrossOriginListeners: true,
      });
      oauthAbortRef.current = cleanup.abort;

      const result = await promise;

      const mcpProvider = allProviders?.["mcp"];
      if (mcpProvider) {
        await mcpProvider.mcpOAuthCallback(result.code, state_token);
      } else {
        const cbRes = await postV2ExchangeOauthCodeForMcpTokens({
          code: result.code,
          state_token,
        });
        if (!(cbRes.status >= 200 && cbRes.status < 300)) {
          const d =
            cbRes.data && typeof cbRes.data === "object" ? cbRes.data : {};
          throw { status: cbRes.status, ...d };
        }
      }

      // Only clear the force-disconnect override AFTER the OAuth dance
      // completes successfully.  Clearing it earlier would let
      // ``liveHasCred=true`` (Reconnect path) render the Connected pill
      // mid-flight, briefly contradicting the in-progress "Reconnecting…"
      // affordance.
      setForceDisconnected(false);
      setConnected(true);
      onSend(retryInstruction ?? "I've connected. Please retry.");
    } catch (e: unknown) {
      const err = e as Record<string, unknown>;
      // Reconnect failures must drop the Connected view so the user sees
      // the error / manual-token input rendered by the not-connected
      // branch.  Setting ``localConnected=false`` alone isn't enough when
      // a stored cred still exists (``liveHasCred=true``) — flip
      // ``forceDisconnected`` so the not-connected branch wins until the
      // user retries.
      setConnected(false);
      setForceDisconnected(true);
      if (err?.status === 400) {
        setShowManualToken(true);
        setError(
          "This server does not support OAuth sign-in. Please enter an API token manually.",
        );
      } else if (
        typeof err?.message === "string" &&
        err.message === "OAuth flow timed out"
      ) {
        setError("OAuth sign-in timed out. Please try again.");
      } else {
        const msg =
          (typeof err?.message === "string" ? err.message : null) ||
          (typeof err?.detail === "string" ? err.detail : null) ||
          "Failed to complete sign-in. Please try again.";
        setError(msg);
      }
    } finally {
      setLoading(false);
      oauthAbortRef.current = null;
    }
  }

  async function handleManualToken() {
    // Re-entrancy guard first — mirrors ``handleConnect`` so both flows
    // present the same shape to readers.  See the comment on
    // ``handleConnect``'s guard for the double-click race this prevents.
    if (loading) return;
    const token = manualToken.trim();
    if (!token) return;
    setLoading(true);
    setError(null);
    try {
      const res = await postV2StoreABearerTokenForAnMcpServer({
        server_url: serverUrl,
        token,
      });
      if (!(res.status >= 200 && res.status < 300))
        throw new Error("Failed to store token");
      // Only clear the force-disconnect override AFTER the API confirms
      // the token was stored.  Clearing it before the await would let
      // ``liveHasCred=true`` (from an existing stale cred) re-render the
      // Connected pill while the request is still in flight, briefly
      // showing a false "Connected" state to the user.
      setForceDisconnected(false);
      setConnected(true);
      onSend(retryInstruction ?? "I've connected. Please retry.");
    } catch (e: unknown) {
      // Keep the force-disconnect override on so the not-connected
      // branch (error banner + manual-token input) stays visible — an
      // existing ``liveHasCred=true`` would otherwise re-render the
      // Connected pill.
      setForceDisconnected(true);
      const err = e as Record<string, unknown>;
      setError(
        (typeof err?.message === "string" ? err.message : null) ||
          "Failed to save token. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  }

  // Already-connected state.  Shown when the backend reports
  // ``has_all_credentials=true`` (model called with
  // ``surface_connect_card=true`` and creds already exist), or after the
  // user just completed a Connect flow in this component instance.  Always
  // expose Reconnect so the user can swap accounts.
  if (connected) {
    // No error banner here — any caught error flips ``forceDisconnected``
    // to true which forces ``connected=false``, so an error inside the
    // connected branch is unreachable.  The not-connected branch below
    // owns the error display.
    return (
      <div className="mt-2 grid gap-2">
        <div className="rounded-lg border border-green-200 bg-green-50 px-3 py-2 text-sm text-green-700">
          Connected to {service}.{" "}
          <button
            type="button"
            onClick={handleConnect}
            disabled={loading}
            className="ml-1 underline underline-offset-2 hover:no-underline disabled:opacity-60"
          >
            {loading ? "Reconnecting…" : "Reconnect"}
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-2 grid gap-2">
      <ContentMessage>{output.message}</ContentMessage>

      <div className="rounded-2xl border bg-background p-4">
        <Button
          variant="primary"
          size="small"
          onClick={handleConnect}
          disabled={loading}
        >
          {loading ? "Connecting…" : `Connect ${service}`}
        </Button>

        {error && (
          <div className="mt-2 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
            {error}
          </div>
        )}

        {showManualToken && (
          <div className="mt-3 flex gap-2">
            <input
              type="password"
              aria-label={`API token for ${service}`}
              placeholder="Paste API token"
              value={manualToken}
              onChange={(e) => setManualToken(e.target.value)}
              onKeyDown={(e) =>
                e.key === "Enter" && !loading && handleManualToken()
              }
              className="flex-1 rounded border px-2 py-1 text-sm"
            />
            <Button
              variant="secondary"
              size="small"
              onClick={handleManualToken}
              disabled={loading || !manualToken.trim()}
            >
              Use Token
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
