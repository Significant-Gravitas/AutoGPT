import { useRef, useState } from "react";
import { Linking } from "react-native";

import type { CredentialMeta } from "@/lib/api/integrations";

import type { AccountChoice } from "./components/AccountPickerSheet";
import type { SheetTarget } from "./components/CredentialSheet/helpers";
import type { PickedCredential } from "./connectedThisRun";
import { credentialLabel, MCP_PROVIDER, webChatURL } from "./helpers";
import type { ServiceAction } from "./listHelpers";
import { startOAuth, type OAuthOutcome } from "./oauth";
import type { FlowOrigin, OAuthTarget } from "./oauthHelpers";

interface Args {
  /** Where sign-ins start from; each one adds the row it answers. */
  origin: Omit<FlowOrigin, "rowKey">;
  onConnected: (rowKey: string, credential: PickedCredential) => void;
}

function toPicked(credential: CredentialMeta): PickedCredential {
  return { id: credential.id, title: credentialLabel(credential) };
}

function tokenNotice(providerName: string, hasOAuth: boolean) {
  return hasOAuth
    ? `Signing in to ${providerName} didn't work. You can paste a token from ${providerName} instead.`
    : `${providerName} doesn't offer a sign-in. Paste a token from ${providerName} instead.`;
}

type MCPTarget = Extract<OAuthTarget, { kind: "mcp" }>;

interface TokenRow {
  notice: string;
  serverURL: string;
  /** The server has a sign-in that failed; closing the sheet offers it again. */
  hasOAuth: boolean;
}

/**
 * Everything a list of services to connect needs: sign-ins in the system
 * browser, the key/login sheet, picking among existing accounts, and the
 * notices those can end in. Rows are told apart by a caller-chosen key.
 */
export function useConnect({ origin, onConnected }: Args) {
  const [busyKey, setBusyKey] = useState<string | null>(null);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [needsPlan, setNeedsPlan] = useState(false);
  const [sheet, setSheet] = useState<SheetTarget | null>(null);
  const [choice, setChoice] = useState<AccountChoice | null>(null);
  // Rows whose MCP sign-in failed: they go straight to the token sheet.
  const [tokens, setTokens] = useState<Record<string, TokenRow>>({});
  // Only one system sign-in sheet can be open at a time.
  const isSigningIn = useRef(false);

  function setError(rowKey: string, message: string | null) {
    setErrors((previous) => {
      const next = { ...previous };
      if (message) next[rowKey] = message;
      else delete next[rowKey];
      return next;
    });
  }

  function openTokenSheet(rowKey: string, target: MCPTarget, notice: string) {
    setError(rowKey, null);
    setSheet({
      rowKey,
      kind: "mcp_token",
      provider: MCP_PROVIDER,
      providerName: target.providerName,
      host: null,
      serverURL: target.serverURL,
      notice,
    });
  }

  /**
   * Whether the row's next press opens the token sheet instead of a sign-in.
   * The same row can be handed another address, which gets a fresh try.
   */
  function usesToken(rowKey: string, serverURL: string | null) {
    return serverURL !== null && tokens[rowKey]?.serverURL === serverURL;
  }

  async function signIn(rowKey: string, target: OAuthTarget) {
    if (target.kind === "mcp" && usesToken(rowKey, target.serverURL)) {
      openTokenSheet(rowKey, target, tokens[rowKey].notice);
      return;
    }
    if (isSigningIn.current) return;
    isSigningIn.current = true;
    setBusyKey(rowKey);
    setError(rowKey, null);
    try {
      handleOutcome(rowKey, target, await startOAuth(target, { ...origin, rowKey }));
    } finally {
      isSigningIn.current = false;
      setBusyKey(null);
    }
  }

  function handleOutcome(rowKey: string, target: OAuthTarget, outcome: OAuthOutcome) {
    if (outcome.status === "connected") {
      onConnected(rowKey, toPicked(outcome.credential));
    } else if (outcome.status === "paywall") {
      setNeedsPlan(true);
    } else if (outcome.status === "use_token" && target.kind === "mcp") {
      const notice = tokenNotice(target.providerName, outcome.hasOAuth);
      setTokens((previous) => ({
        ...previous,
        [rowKey]: { notice, serverURL: target.serverURL, hasOAuth: outcome.hasOAuth },
      }));
      openTokenSheet(rowKey, target, notice);
    } else if (outcome.status === "failed") {
      setError(rowKey, outcome.message);
    }
  }

  function openSheet(target: SheetTarget) {
    setError(target.rowKey, null);
    setSheet(target);
  }

  function closeSheet() {
    // Closed without a token: a server that has a sign-in gets another try.
    if (sheet?.kind === "mcp_token" && tokens[sheet.rowKey]?.hasOAuth) {
      const { rowKey } = sheet;
      setTokens((previous) => {
        const next = { ...previous };
        delete next[rowKey];
        return next;
      });
    }
    setSheet(null);
  }

  function handleSaved(rowKey: string, credential: CredentialMeta) {
    // A save that ends after another sheet opened leaves that one alone.
    setSheet((open) => (open?.rowKey === rowKey ? null : open));
    onConnected(rowKey, toPicked(credential));
  }

  function openChoice(next: AccountChoice) {
    setError(next.rowKey, null);
    setChoice(next);
  }

  function closeChoice() {
    setChoice(null);
  }

  function handlePick(rowKey: string, account: { id: string; title: string }) {
    setChoice(null);
    onConnected(rowKey, { id: account.id, title: account.title });
  }

  function openOnWeb() {
    Linking.openURL(webChatURL(origin.sessionID)).catch(() => undefined);
  }

  /** Starts whatever a service's row offers: a sign-in, a sheet, or the web. */
  function connectService(rowKey: string, providerName: string, action: ServiceAction) {
    switch (action.kind) {
      case "oauth":
        void signIn(rowKey, { kind: "provider", provider: action.provider, providerName, scopes: [] });
        return;
      case "mcp":
        void signIn(rowKey, { kind: "mcp", serverURL: action.serverURL, providerName });
        return;
      case "sheet":
        openSheet({ rowKey, kind: action.sheet, provider: action.provider, providerName, host: null, serverURL: null });
        return;
      case "mcp_token":
        openSheet({
          rowKey,
          kind: "mcp_token",
          provider: MCP_PROVIDER,
          providerName,
          host: null,
          serverURL: action.serverURL,
        });
        return;
      case "web":
        openOnWeb();
    }
  }

  return {
    busyKey,
    errors,
    usesToken,
    needsPlan,
    signIn,
    sheet,
    openSheet,
    closeSheet,
    handleSaved,
    choice,
    openChoice,
    closeChoice,
    handlePick,
    openOnWeb,
    connectService,
    /** Shows a failure on a row, e.g. a grant that came after the sign-in. */
    showError: setError,
  };
}
