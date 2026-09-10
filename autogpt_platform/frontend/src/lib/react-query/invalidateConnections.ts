import type { QueryClient } from "@tanstack/react-query";

import {
  getGetV2ListChatConnectionsQueryKey,
  getGetV2ListChatTransportsQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";

/**
 * Refresh everything a connect or disconnect changes.
 *
 * A credential is not just a row in the integrations list. The server also
 * derives the chat connection offers and the transport list from it, and those
 * are what the AI subscriptions section and the copilot's connection picker
 * render. Invalidating the credentials list alone leaves both showing the
 * state from before the sign-in for as long as they stay fresh -- a minute,
 * under the client's default `staleTime` -- so returning from a ChatGPT device
 * login reads as if the sign-in did nothing until the page is reloaded.
 *
 * Every path that creates or removes a credential should call this rather than
 * pick a subset: which lists a given provider feeds is a server-side fact, and
 * a caller that guesses gets it wrong the moment the server adds one.
 */
export async function invalidateConnectionQueries(
  queryClient: QueryClient,
): Promise<void> {
  await Promise.all([
    queryClient.invalidateQueries({
      queryKey: getGetV1ListCredentialsQueryKey(),
    }),
    queryClient.invalidateQueries({
      queryKey: getGetV2ListChatConnectionsQueryKey(),
    }),
    queryClient.invalidateQueries({
      queryKey: getGetV2ListChatTransportsQueryKey(),
    }),
  ]);
}
