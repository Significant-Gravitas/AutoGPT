import { QueryClient } from "@tanstack/react-query";
import { describe, expect, it, vi } from "vitest";

import {
  getGetV2ListChatConnectionsQueryKey,
  getGetV2ListChatTransportsQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { getGetV1ListCredentialsQueryKey } from "@/app/api/__generated__/endpoints/integrations/integrations";

import { invalidateConnectionQueries } from "../invalidateConnections";

function hash(key: readonly unknown[]) {
  return JSON.stringify(key);
}

describe("invalidateConnectionQueries", () => {
  it("refreshes the credentials, chat connections and transports lists", async () => {
    const client = new QueryClient();
    const invalidateQueries = vi.fn().mockResolvedValue(undefined);
    client.invalidateQueries = invalidateQueries;

    await invalidateConnectionQueries(client);

    const invalidated = invalidateQueries.mock.calls.map(([args]) =>
      hash(args.queryKey),
    );
    expect(invalidated).toContain(hash(getGetV1ListCredentialsQueryKey()));
    expect(invalidated).toContain(hash(getGetV2ListChatConnectionsQueryKey()));
    expect(invalidated).toContain(hash(getGetV2ListChatTransportsQueryKey()));
  });

  it("waits for every invalidation before resolving", async () => {
    const client = new QueryClient();
    let settled = 0;
    client.invalidateQueries = vi.fn(
      () =>
        new Promise<void>((resolve) =>
          setTimeout(() => {
            settled += 1;
            resolve();
          }, 0),
        ),
    );

    await invalidateConnectionQueries(client);

    expect(settled).toBe(3);
  });
});
