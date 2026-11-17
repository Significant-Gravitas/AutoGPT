"use client";

import { useGetExperimentalGetSessionExecutor } from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { ExecutorStatus } from "@/app/api/__generated__/models/executorStatus";
import type { RecordingStartRequestChannelsItem } from "@/app/api/__generated__/models/recordingStartRequestChannelsItem";
import type { RecordingStartRequestInterpretationRoute } from "@/app/api/__generated__/models/recordingStartRequestInterpretationRoute";

export type LocalPCExecutorStatus = Omit<
  ExecutorStatus,
  "recording_routes" | "recording_channels"
> & {
  recording_routes?: RecordingStartRequestInterpretationRoute[] | null;
  recording_channels?: RecordingStartRequestChannelsItem[] | null;
};

/**
 * Poll-once-per-15s view of the shim's HELLO metadata for the active
 * copilot session. Returns `{kind: "none"}` immediately when the shim
 * isn't connected — the badge falls back to the static "Local PC mode"
 * label in that case.
 *
 * Disabled (no fetch) when `sessionId` is missing or `enabled === false`.
 */
export function useLocalPCExecutor(
  sessionID: string | null,
  options: { enabled?: boolean } = {},
) {
  const enabled = (options.enabled ?? true) && !!sessionID;

  return useGetExperimentalGetSessionExecutor<LocalPCExecutorStatus>(
    sessionID ?? "",
    {
      query: {
        enabled,
        refetchInterval: 15_000,
        refetchOnWindowFocus: false,
        staleTime: 10_000,
        select: (response) => response.data as LocalPCExecutorStatus,
      },
    },
  );
}
