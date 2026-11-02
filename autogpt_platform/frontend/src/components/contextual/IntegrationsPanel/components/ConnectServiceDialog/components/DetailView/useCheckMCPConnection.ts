"use client";

import { useState } from "react";
import { postV2DiscoverAvailableToolsOnAnMcpServer } from "@/app/api/__generated__/endpoints/mcp/mcp";
import { getAPIResponseError, getErrorMessage } from "@/lib/mcp-errors";

export function useCheckMCPConnection(serverURL: string) {
  const [isPending, setIsPending] = useState(false);
  const [toolCount, setToolCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function check() {
    setIsPending(true);
    setError(null);
    setToolCount(null);
    try {
      const response = await postV2DiscoverAvailableToolsOnAnMcpServer({
        server_url: serverURL,
        use_saved_credentials: false,
      });
      if (response.status !== 200)
        throw getAPIResponseError(response.status, response.data);
      setToolCount(response.data.tools.length);
    } catch (error) {
      setError(getErrorMessage(error));
    } finally {
      setIsPending(false);
    }
  }

  return { check, isPending, toolCount, error };
}
