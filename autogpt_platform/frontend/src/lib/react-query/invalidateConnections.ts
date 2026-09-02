import type { QueryClient } from "@tanstack/react-query";

import {
  getGetV2ListChatConnectionsQueryKey,
  getGetV2ListChatTransportsQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";

/** Refresh every client-side view derived from a credential change. */
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
