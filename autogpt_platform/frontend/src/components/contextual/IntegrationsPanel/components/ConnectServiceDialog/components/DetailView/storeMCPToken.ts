import {
  postV2DiscoverAvailableToolsOnAnMcpServer,
  postV2StoreABearerTokenForAnMcpServer,
} from "@/app/api/__generated__/endpoints/mcp/mcp";
import { prepareMCPAuthCredential, type MCPAuthScheme } from "@/lib/mcp-auth";
import { getAPIResponseError } from "@/lib/mcp-errors";

export async function storeMCPToken(
  serverURL: string,
  token: string,
  scheme: MCPAuthScheme,
  signal: AbortSignal,
) {
  signal.throwIfAborted();
  const authValue = prepareMCPAuthCredential(token, scheme);
  const probe = await postV2DiscoverAvailableToolsOnAnMcpServer(
    {
      server_url: serverURL,
      auth_token: authValue,
    },
    { signal },
  );
  signal.throwIfAborted();
  if (probe.status !== 200) throw getAPIResponseError(probe.status, probe.data);
  const stored = await postV2StoreABearerTokenForAnMcpServer(
    {
      server_url: serverURL,
      token: authValue,
    },
    { signal },
  );
  signal.throwIfAborted();
  if (stored.status !== 200)
    throw getAPIResponseError(stored.status, stored.data);
  return stored.data;
}
