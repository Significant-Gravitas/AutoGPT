"use client";

import { useGetExperimentalGetSessionExecutor } from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { ExecutorStatus } from "@/app/api/__generated__/models/executorStatus";
import type { RecordingStartRequestChannelsItem } from "@/app/api/__generated__/models/recordingStartRequestChannelsItem";
import type { RecordingStartRequestInterpretationRoute } from "@/app/api/__generated__/models/recordingStartRequestInterpretationRoute";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";

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
 *
 * Also disabled, with `isTurnedOff`, once the flag service says this user is
 * outside the Local PC rollout: the route answers them 404. A flag that has
 * not answered, or cannot, is not an "off": the request goes out, so a flag
 * outage does not take the status off a chat that runs on the person's
 * computer.
 */
export function useLocalPCExecutor(
  sessionID: string | null,
  options: { enabled?: boolean } = {},
) {
  const rollout = useFlagStatus(Flag.LOCAL_PC_EXECUTOR);
  const isTurnedOff = rollout.answered && !rollout.enabled;
  const isWanted = (options.enabled ?? true) && !!sessionID;

  const query = useGetExperimentalGetSessionExecutor<LocalPCExecutorStatus>(
    sessionID ?? "",
    {
      query: {
        enabled: isWanted && rollout.ready && !isTurnedOff,
        refetchInterval: 15_000,
        refetchOnWindowFocus: false,
        staleTime: 10_000,
        select: (response) => response.data as LocalPCExecutorStatus,
      },
    },
  );

  return {
    data: isTurnedOff ? undefined : query.data,
    isError: !isTurnedOff && query.isError,
    isLoading: query.isLoading || (isWanted && !rollout.ready),
    isSuccess: !isTurnedOff && query.isSuccess,
    isTurnedOff,
    refetch: query.refetch,
  };
}
