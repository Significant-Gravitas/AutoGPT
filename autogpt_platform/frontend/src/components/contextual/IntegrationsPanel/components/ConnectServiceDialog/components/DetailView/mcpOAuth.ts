import {
  postV2ExchangeOauthCodeForMcpTokens,
  postV2InitiateOauthLoginForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { getAPIResponseError, getErrorStatus } from "@/lib/mcp-errors";
import { openOAuthPopup } from "@/lib/oauth-popup";

interface Args {
  serverURL: string;
  onPopup: (abort: (reason?: string) => void) => void;
}

export async function connectMCPOAuth({ serverURL, onPopup }: Args) {
  let login;
  try {
    login = await postV2InitiateOauthLoginForAnMcpServer({
      server_url: serverURL,
    });
    if (login.status !== 200)
      throw getAPIResponseError(login.status, login.data);
  } catch (error) {
    if (getErrorStatus(error) === 400) return null;
    throw error;
  }
  const { login_url, state_token } = login.data;
  const { promise, cleanup } = openOAuthPopup(login_url, {
    stateToken: state_token,
    useCrossOriginListeners: true,
  });
  onPopup(cleanup.abort);
  const result = await promise;
  const exchanged = await postV2ExchangeOauthCodeForMcpTokens({
    code: result.code,
    state_token,
    iss: result.iss,
  });
  if (exchanged.status !== 200)
    throw getAPIResponseError(exchanged.status, exchanged.data);
  return exchanged.data;
}
