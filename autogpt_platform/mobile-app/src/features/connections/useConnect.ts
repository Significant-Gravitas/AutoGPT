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

  async function signIn(rowKey: string, target: OAuthTarget) {
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
    } else if (outcome.status === "no_oauth" && target.kind === "mcp") {
      setSheet({
        rowKey,
        kind: "mcp_token",
        provider: "mcp",
        providerName: target.providerName,
        host: null,
        serverURL: target.serverURL,
      });
    } else if (outcome.status === "failed" || outcome.status === "no_oauth") {
      setError(rowKey, outcome.message);
    }
  }

  function openSheet(target: SheetTarget) {
    setError(target.rowKey, null);
    setSheet(target);
  }

  function closeSheet() {
    setSheet(null);
  }

  function handleSaved(rowKey: string, credential: CredentialMeta) {
    setSheet(null);
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
