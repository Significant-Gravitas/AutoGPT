import {
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { getAPIResponseError, getErrorStatus } from "@/lib/mcp-errors";
import { openOAuthPopup } from "@/lib/oauth-popup";

interface Args {
  serverURL: string;
  scopes?: string[];
  signal: AbortSignal;
}

export async function connectMCPOAuth({ serverURL, scopes, signal }: Args) {
  signal.throwIfAborted();
  let login;
  try {
    login = await postV2InitiateOauthLoginForAnMcpServer(
      {
        server_url: serverURL,
        ...(scopes === undefined ? {} : { scopes }),
      },
      { signal },
    );
    signal.throwIfAborted();
    if (login.status !== 200)
      throw getAPIResponseError(login.status, login.data);
  } catch (error) {
    signal.throwIfAborted();
    if (getErrorStatus(error) === 400) return null;
    throw error;
  }
  signal.throwIfAborted();
  const { login_url, state_token } = login.data;
  const { promise, cleanup } = openOAuthPopup(login_url, {
    stateToken: state_token,
    useCrossOriginListeners: true,
  });
  function abortPopup() {
    cleanup.abort();
  }
  signal.addEventListener("abort", abortPopup, { once: true });
  try {
    if (signal.aborted) abortPopup();
    signal.throwIfAborted();
    const result = await promise;
    signal.throwIfAborted();
    const exchanged = await postV2ExchangeOauthCodeForMcpTokens(
      {
        code: result.code,
        state_token,
        iss: result.iss,
      },
      { signal },
    );
    signal.throwIfAborted();
    if (exchanged.status !== 200)
      throw getAPIResponseError(exchanged.status, exchanged.data);
    return exchanged.data;
  } finally {
    signal.removeEventListener("abort", abortPopup);
  }
}
