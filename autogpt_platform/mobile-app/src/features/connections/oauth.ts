import * as WebBrowser from "expo-web-browser";

import { ApiError } from "@/lib/api/client";
import {
  finishMCPLogin,
  finishProviderLogin,
  integrationKeys,
  MCP_NO_OAUTH,
  OAUTH_RETURN_URL,
  startMCPLogin,
  startProviderLogin,
  type CredentialMeta,
} from "@/lib/api/integrations";
import { queryClient, sessionGeneration } from "@/lib/session";

import {
  parseOAuthReturn,
  providerErrorMessage,
  type FlowOrigin,
  type OAuthReturn,
  type OAuthTarget,
  type PendingFlow,
} from "./oauthHelpers";
import { clearPendingFlow, readPendingFlow, savePendingFlow } from "./pendingFlow";

export type OAuthOutcome =
  | { status: "connected"; credential: CredentialMeta }
  | { status: "cancelled" }
  /**
   * An MCP server's sign-in can't start, so it takes a token instead.
   * `hasOAuth` is false when the server offers no sign-in at all.
   */
  | { status: "use_token"; hasOAuth: boolean }
  /** The account needs a plan, which only the web sells. */
  | { status: "paywall" }
  | { status: "failed"; message: string };

export interface FinishedFlow {
  outcome: OAuthOutcome;
  flow: PendingFlow | null;
}

// Flows started in this app run. On Android the return can reach the
// deep-link route after the browser session already gave up, so they stay.
const liveFlows = new Map<string, PendingFlow>();
// One code swap per state, whoever asks first: the session or the route.
const finishing = new Map<string, Promise<FinishedFlow>>();
// Sign-ins whose result someone already acted on.
const claimed = new Set<string>();

function failed(message: string): OAuthOutcome {
  return { status: "failed", message };
}

/** Signs in in the system browser and saves the account it comes back with. */
export async function startOAuth(
  target: OAuthTarget,
  origin: FlowOrigin,
): Promise<OAuthOutcome> {
  let login: { loginURL: string; state: string };
  try {
    login = await requestLogin(target);
  } catch (error) {
    return loginFailure(error, target);
  }
  const flow: PendingFlow = { ...origin, state: login.state, target, savedAt: Date.now() };
  liveFlows.set(flow.state, flow);
  // Saved before the browser opens: Android may kill the app meanwhile.
  await savePendingFlow(flow).catch(() => undefined);
  let result: WebBrowser.WebBrowserAuthSessionResult;
  try {
    result = await WebBrowser.openAuthSessionAsync(login.loginURL, OAUTH_RETURN_URL);
  } catch {
    void clearPendingFlow(flow.state);
    return failed("Couldn't open the sign-in page. Try again.");
  }
  // From here this session acts on the result; a later link for it is a repeat.
  claimed.add(flow.state);
  const returned = result.type === "success" ? parseOAuthReturn(result.url) : null;
  if (!returned) {
    const elsewhere = finishing.get(flow.state);
    if (elsewhere) return (await elsewhere).outcome;
    void clearPendingFlow(flow.state);
    return { status: "cancelled" };
  }
  if (returned.state !== flow.state) {
    void clearPendingFlow(flow.state);
    return failed("That sign-in didn't match this one. Try again.");
  }
  return (await completeOAuthReturn(returned)).outcome;
}

/** Swaps the returned code for an account, once per sign-in. */
export function completeOAuthReturn(returned: OAuthReturn) {
  const running = finishing.get(returned.state);
  if (running) return running;
  const promise = finish(returned);
  finishing.set(returned.state, promise);
  return promise;
}

/**
 * True for the first caller to act on this sign-in's result. A link the
 * system delivers twice must not grant access again after the user took it away.
 */
export function claimOAuthResult(state: string) {
  if (claimed.has(state)) return false;
  claimed.add(state);
  return true;
}

/** Puts a new account in the cached list at once, then checks with the server. */
export function rememberCredential(credential: CredentialMeta) {
  queryClient.setQueryData<CredentialMeta[]>(integrationKeys.credentials, (old) =>
    old ? [...old.filter((item) => item.id !== credential.id), credential] : old,
  );
  void queryClient.invalidateQueries({ queryKey: integrationKeys.credentials });
}

async function requestLogin(target: OAuthTarget) {
  if (target.kind === "mcp") return startMCPLogin(target.serverURL);
  const options = { scopes: target.scopes, credentialID: target.credentialID };
  try {
    return await startProviderLogin(target.provider, options);
  } catch (error) {
    // The account to widen may be gone by now; a fresh sign-in still works.
    if (!target.credentialID || !isMissingUpgradeTarget(error)) throw error;
    return startProviderLogin(target.provider, { scopes: target.scopes });
  }
}

function isMissingUpgradeTarget(error: unknown) {
  return (
    error instanceof ApiError &&
    error.status === 404 &&
    error.message.toLowerCase().includes("upgrade")
  );
}

function loginFailure(error: unknown, target: OAuthTarget): OAuthOutcome {
  if (error instanceof ApiError && error.status === 402) return { status: "paywall" };
  // The backend's 400 text is written for developers, so it is never shown.
  if (target.kind === "mcp" && error instanceof ApiError && error.status === 400) {
    return { status: "use_token", hasOAuth: error.code !== MCP_NO_OAUTH };
  }
  return failed(`Couldn't start signing in to ${target.providerName}. Try again.`);
}

async function finish(returned: OAuthReturn): Promise<FinishedFlow> {
  const flow =
    liveFlows.get(returned.state) ?? (await readPendingFlow(returned.state));
  if (!flow) {
    return {
      flow: null,
      outcome: failed("This sign-in has expired. Start it again from the chat."),
    };
  }
  const outcome = await exchange(returned, flow.target);
  // Kept until the swap ends, so an app killed during it can still finish
  // from the same link.
  void clearPendingFlow(flow.state);
  return { flow, outcome };
}

async function exchange(returned: OAuthReturn, target: OAuthTarget): Promise<OAuthOutcome> {
  if (returned.error || !returned.code) {
    const message = providerErrorMessage(
      returned.error ?? "missing_code",
      target.providerName,
    );
    return message ? failed(message) : { status: "cancelled" };
  }
  const startedIn = sessionGeneration();
  try {
    const credential =
      target.kind === "mcp"
        ? await finishMCPLogin(returned.code, returned.state, returned.iss)
        : await finishProviderLogin(target.provider, returned.code, returned.state);
    // Signed out meanwhile: the account belongs to whoever started this.
    if (startedIn !== sessionGeneration()) return { status: "cancelled" };
    rememberCredential(credential);
    return { status: "connected", credential };
  } catch (error) {
    if (error instanceof ApiError && error.status === 402) return { status: "paywall" };
    return failed(`Couldn't connect ${target.providerName}. Try again.`);
  }
}
