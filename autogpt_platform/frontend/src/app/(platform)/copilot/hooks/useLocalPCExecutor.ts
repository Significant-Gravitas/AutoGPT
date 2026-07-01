"use client";

import { useQuery } from "@tanstack/react-query";
import { customMutator } from "@/app/api/mutators/custom-mutator";

/**
 * Shape of `GET /api/copilot/sessions/{session_id}/executor`.
 *
 * Mirrors `backend/api/features/local_executor/routes.py:ExecutorStatus`.
 * When `kind === "none"`, every other field is null — no shim is connected
 * for this session on the worker we hit.
 *
 * **Todo**: regenerate the typed client via `pnpm generate:api` once the
 * OpenAPI snapshot is updated. Until then this hook keeps the contract
 * type-safe via the inline interface below.
 */
export interface ExecutorStatus {
  kind: "shim" | "none";
  platform: string | null;
  arch: string | null;
  allowed_root: string | null;
  machine_id: string | null;
  shim_version: string | null;
  capabilities: string[] | null;
  computer_use_features: string[] | null;
}

/**
 * Poll-once-per-15s view of the shim's HELLO metadata for the active
 * copilot session. Returns `{kind: "none"}` immediately when the shim
 * isn't connected — the badge falls back to the static "Local PC mode"
 * label in that case.
 *
 * Disabled (no fetch) when `sessionId` is missing or `enabled === false`.
 */
export function useLocalPCExecutor(
  sessionId: string | null,
  options: { enabled?: boolean } = {},
) {
  const enabled = (options.enabled ?? true) && !!sessionId;

  return useQuery<ExecutorStatus>({
    queryKey: ["local-pc-executor", sessionId],
    enabled,
    refetchInterval: 15_000,
    refetchOnWindowFocus: false,
    staleTime: 10_000,
    queryFn: async () => {
      const resp = await customMutator<{
        data: ExecutorStatus;
        status: number;
        headers: Headers;
      }>(`/api/copilot/sessions/${sessionId}/executor`, { method: "GET" });
      return resp.data;
    },
  });
}
