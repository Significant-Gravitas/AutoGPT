import { useQuery } from "@tanstack/react-query";

import {
  claimFollowUp,
  markConnected,
  settleFollowUp,
  useCardRun,
} from "@/features/connections/connectedThisRun";
import {
  logoID,
  MCP_PROVIDER,
  requestHost,
  requestServerURL,
  type CredentialRequest,
} from "@/features/connections/helpers";
import { useConnect } from "@/features/connections/useConnect";
import { ApiError } from "@/lib/api/client";
import {
  grantExpertCredentials,
  integrationKeys,
  listCredentials,
  listExpertCredentials,
  recordCredentialPicks,
} from "@/lib/api/integrations";
import { queryClient } from "@/lib/session";

import {
  buildSelections,
  cardIntro,
  displayName,
  followUpMessage,
  isMCPSetup,
  needsManualContinue,
  rowStatus,
  type ConnectedRow,
  type SetupRequest,
} from "./helpers";

const LIST_STALE_MS = 10_000;
const SEND_TRIES = 10;
const SEND_RETRY_MS = 500;
const GRANT_PAYWALL_MESSAGE = "Allowing access needs an active plan. Finish this on the web.";

interface Args {
  sessionID: string;
  toolCallID: string;
  setup: SetupRequest;
  replyText: string | null;
  isBusy: boolean;
  onSend: (text: string) => boolean;
}

export function useSetupCard({ sessionID, toolCallID, setup, replyText, isBusy, onSend }: Args) {
  const cardKey = `${sessionID}:${toolCallID}`;
  const run = useCardRun(cardKey);
  const expertID =
    setup.requests.find((request) => request.expertGrant)?.expertGrant?.expertID ?? null;
  const credentials = useQuery({
    queryKey: integrationKeys.credentials,
    queryFn: listCredentials,
    staleTime: LIST_STALE_MS,
  });
  const granted = useQuery({
    queryKey: integrationKeys.expertCredentials(expertID ?? ""),
    queryFn: () => listExpertCredentials(expertID ?? ""),
    enabled: expertID !== null,
    staleTime: LIST_STALE_MS,
  });
  const connect = useConnect({
    origin: { sessionID, cardKey, expertID },
    onConnected: (rowKey, credential) => markConnected(cardKey, rowKey, credential),
  });

  const grantedIDs = granted.data?.map((item) => item.credentialID) ?? null;
  const rows = setup.requests.map((request) => {
    const status = rowStatus(request, {
      pick: run.picks[request.key],
      credentials: credentials.data ?? [],
      grantedIDs,
      rejection: setup.rejection,
    });
    const serverURL = requestServerURL(request, setup.serverURL);
    const signsIn = status.kind === "reconnect" || status.kind === "connect";
    return {
      request,
      name: displayName(request, setup),
      logoID: logoID(request.provider, serverURL),
      status,
      // A failed MCP sign-in turns the row's next press into the token sheet.
      usesToken:
        signsIn && request.provider === MCP_PROVIDER && connect.usesToken(request.key, serverURL),
    };
  });
  const connectedRows: ConnectedRow[] = rows.flatMap(({ request, status }) =>
    status.kind === "connected" ? [{ request, credentialID: status.credentialID }] : [],
  );
  const isAllConnected = connectedRows.length === rows.length;
  const hasPicks = setup.requests.some((request) => run.picks[request.key]);
  const isLoading = credentials.isPending || (expertID !== null && granted.isPending);
  const canContinue =
    isAllConnected && run.followUp === "idle" && !setup.needsInputs && (hasPicks || replyText === null);
  // A sign-in from this card in this app run sends the reply by itself; a
  // card whose accounts were already there, or that the chat has moved past,
  // waits for Continue.
  const shouldAutoSend =
    canContinue &&
    hasPicks &&
    replyText === null &&
    !run.error &&
    !isBusy &&
    !needsManualContinue(setup);
  const needsWeb =
    setup.needsInputs ||
    connect.needsPlan ||
    run.error === GRANT_PAYWALL_MESSAGE ||
    rows.some(({ status }) => status.kind === "connect" && status.method === "web");

  async function grantNewAccounts() {
    if (!expertID) return true;
    const ids = connectedRows
      .filter(({ request, credentialID }) => request.expertGrant && !grantedIDs?.includes(credentialID))
      .map(({ credentialID }) => credentialID);
    if (ids.length === 0) return true;
    try {
      await grantExpertCredentials(expertID, [...new Set(ids)]);
    } catch (error) {
      const isPaywall = error instanceof ApiError && error.status === 402;
      settleFollowUp(
        cardKey,
        false,
        isPaywall ? GRANT_PAYWALL_MESSAGE : "Couldn't allow access. Try again.",
      );
      return false;
    }
    void queryClient.invalidateQueries({ queryKey: integrationKeys.expertCredentials(expertID) });
    return true;
  }

  async function sendFollowUp() {
    if (!claimFollowUp(cardKey)) return;
    if (!(await grantNewAccounts())) return;
    const selections = buildSelections(connectedRows);
    if (!isMCPSetup(setup) && Object.keys(selections).length > 0) {
      // An account recorded earlier in the chat would win over this pick, so
      // the chat only goes on once the pick is saved.
      try {
        await recordCredentialPicks(sessionID, selections);
      } catch {
        settleFollowUp(cardKey, false, "Couldn't save which account to use. Try again.");
        return;
      }
    }
    // Refused while the chat is busy: the card then offers Try again.
    const isSent = await sendWhenReady(onSend, followUpMessage(setup, connectedRows));
    settleFollowUp(cardKey, isSent, isSent ? null : "Couldn't continue yet. Try again.");
  }

  function signInFor(request: CredentialRequest, name: string, credentialID?: string) {
    if (request.provider === "mcp") {
      const serverURL = requestServerURL(request, setup.serverURL);
      if (serverURL) void connect.signIn(request.key, { kind: "mcp", serverURL, providerName: name });
      else connect.openOnWeb();
      return;
    }
    void connect.signIn(request.key, {
      kind: "provider",
      provider: request.provider,
      providerName: name,
      scopes: request.scopes,
      credentialID,
    });
  }

  function handleRow(index: number) {
    const { request, name, logoID: rowLogoID, status } = rows[index];
    if (status.kind === "allow" || status.kind === "choose") {
      if (status.kind === "allow" && status.accounts.length === 1) {
        markConnected(cardKey, request.key, status.accounts[0]);
        return;
      }
      connect.openChoice({
        rowKey: request.key,
        providerName: name,
        logoID: rowLogoID,
        mode: status.kind,
        accounts: status.accounts,
      });
      return;
    }
    if (status.kind === "reconnect") {
      signInFor(request, name, status.credentialID);
      return;
    }
    if (status.kind !== "connect") return;
    const { method } = status;
    if (method === "oauth" || method === "mcp") signInFor(request, name);
    else if (method === "web") connect.openOnWeb();
    else {
      connect.openSheet({
        rowKey: request.key,
        kind: method,
        provider: request.provider,
        providerName: name,
        host: requestHost(request),
        serverURL: null,
      });
    }
  }

  function handleContinue() {
    void sendFollowUp();
  }

  return {
    rows,
    intro: cardIntro(rows),
    isLoading,
    run,
    connect,
    canContinue,
    shouldAutoSend,
    needsWeb,
    isAnyBusy: connect.busyKey !== null,
    handleRow,
    handleContinue,
    sendFollowUp,
  };
}

// Coming back from the sign-in browser starts a sync, and the chat refuses a
// send until it ends, so give it a few seconds before asking for Try again.
async function sendWhenReady(onSend: (text: string) => boolean, text: string) {
  for (let attempt = 1; attempt <= SEND_TRIES; attempt += 1) {
    if (onSend(text)) return true;
    if (attempt < SEND_TRIES) await new Promise((resolve) => setTimeout(resolve, SEND_RETRY_MS));
  }
  return false;
}
