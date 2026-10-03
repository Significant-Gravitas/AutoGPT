import {
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import {
  getAPIResponseError,
  getErrorCode,
  getErrorMessage,
  getErrorStatus,
} from "@/lib/mcp-errors";
import {
  OAUTH_ERROR_WINDOW_CLOSED,
  openOAuthPopup,
  preOpenOAuthPopup,
} from "@/lib/oauth-popup";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

interface Args {
  serverURL: string;
  scopes?: string[];
  signal: AbortSignal;
  exchange?: (
    code: string,
    state: string,
    iss?: string,
  ) => Promise<
    Pick<CredentialsMetaResponse, "id" | "provider" | "type"> &
      Partial<CredentialsMetaResponse>
  >;
}

/** Mirrors `NO_OAUTH_CODE` in `backend/api/features/mcp/routes.py`. */
export const NO_OAUTH_CODE = "no_oauth";

export async function connectMCPOAuth({
  serverURL,
  scopes,
  signal,
  exchange,
}: Args) {
  signal.throwIfAborted();
  const preOpenedWindow = preOpenOAuthPopup();
  let ownsBlankWindow = true;
  let abortPopup: (() => void) | undefined;
  function cancel() {
    if (abortPopup) abortPopup();
    else if (ownsBlankWindow) {
      ownsBlankWindow = false;
      if (preOpenedWindow && !preOpenedWindow.closed) preOpenedWindow.close();
    }
  }
  signal.addEventListener("abort", cancel, { once: true });
  try {
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
      // The login route writes eight different 400s, each explaining a
      // different failure. Returning a bare null threw all of them away and
      // told the user to find an API token, even when the real cause was a
      // failed client registration on a server that does support OAuth. Hand
      // the reason back, with the route's own verdict on whether this server
      // has any OAuth to offer, so the caller can show one and branch on the
      // other instead of reading the prose.
      if (getErrorStatus(error) === 400)
        return {
          reason: getErrorMessage(error),
          noOAuth: getErrorCode(error) === NO_OAUTH_CODE,
        } as const;
      throw error;
    }
    signal.throwIfAborted();
    const { login_url, state_token } = login.data;
    if (preOpenedWindow?.closed) throw new Error(OAUTH_ERROR_WINDOW_CLOSED);
    const { promise, cleanup } = openOAuthPopup(login_url, {
      stateToken: state_token,
      preOpenedWindow,
      useCrossOriginListeners: true,
    });
    abortPopup = function () {
      cleanup.abort();
    };
    if (signal.aborted) cancel();
    signal.throwIfAborted();
    const result = await promise;
    signal.throwIfAborted();
    if (exchange) {
      const credential = await exchange(result.code, state_token, result.iss);
      signal.throwIfAborted();
      return {
        ...credential,
        title: credential.title ?? null,
        scopes: credential.scopes ?? null,
        username: credential.username ?? null,
      };
    }
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
    signal.removeEventListener("abort", cancel);
    if (!abortPopup) cancel();
  }
}
