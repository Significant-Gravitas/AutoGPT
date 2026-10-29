import { useFinalizeBrainDump } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { FinalizeResponse } from "@/app/api/__generated__/models/finalizeResponse";
import { useState } from "react";
import { describeError, type RecordingSnapshot } from "./helpers";

// Nothing persists the finalize response from the real onboarding flow, so
// the only way this page can show one is to issue its own call.
export function useDebugFinalize(snapshot: RecordingSnapshot) {
  const [response, setResponse] = useState<FinalizeResponse | null>(null);
  const [roundTripMs, setRoundTripMs] = useState<number | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const { mutateAsync, isPending } = useFinalizeBrainDump();

  const meta = snapshot.meta;

  async function run() {
    if (!meta) return;
    setErrorMessage(null);
    const startedAt = performance.now();
    try {
      const result = await mutateAsync({
        data: {
          recording_id: meta.recordingId,
          input_mode: "voice",
          duration_secs: meta.durationSecs,
          mime_type: meta.mimeType,
        },
      });
      setRoundTripMs(performance.now() - startedAt);
      setResponse(result.status === 200 ? result.data : null);
    } catch (error) {
      setRoundTripMs(performance.now() - startedAt);
      setResponse(null);
      setErrorMessage(describeError(error));
    }
  }

  return {
    run,
    response,
    roundTripMs,
    errorMessage,
    isRunning: isPending,
    canRun: meta !== null && !isPending,
    recordingId: meta?.recordingId ?? null,
  };
}
