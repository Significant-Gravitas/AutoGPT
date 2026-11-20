import { useRef, useState } from "react";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import type { useBrainDumpRecorder } from "./useBrainDumpRecorder";
import {
  type MicrophonePermission,
  useMicrophonePermission,
} from "./useMicrophonePermission";

type Recorder = Pick<
  ReturnType<typeof useBrainDumpRecorder>,
  "permissionDenied" | "clearPermissionDenied" | "start"
>;

type MicRetryVia = "retry_link" | "record_instead";

interface Args {
  recorder: Recorder;
  // Whether the user still wants the take once the prompt is answered: the
  // typed text may have been sent while it was open.
  canRecord: () => boolean;
  // The retry opened the microphone and the take is running.
  onRecording: () => void;
}

// The way back to Talk once the browser refused the microphone. Chrome
// refuses with the same error whether the prompt was blocked or only
// closed, and a block is one click to undo by the address bar, so neither
// is treated as final.
export function useMicrophoneRetry({ recorder, canRecord, onRecording }: Args) {
  const permission = useMicrophonePermission({
    onChange: handlePermissionChange,
  });
  // A retry clears the refusal while the browser is asked again; the
  // notice stays up through it and is refreshed in place if it is refused.
  // Only a retry after a refusal keeps the blocked notice up; one after an
  // unavailable mic keeps that notice instead.
  const [isRetryingRefusal, setIsRetryingRefusal] = useState(false);
  const isRetryingRef = useRef(false);

  function handlePermissionChange(state: MicrophonePermission) {
    if (!recorder.permissionDenied) return;
    if (state !== "granted" && state !== "prompt") return;
    recorder.clearPermissionDenied();
    if (state !== "granted") return;
    trackBrainDump("brain_dump_permission_retried", {
      result: "granted",
      permission_state: state,
      via: "permission_change",
    });
  }

  async function retry(via: MicRetryVia) {
    if (isRetryingRef.current) return;
    isRetryingRef.current = true;
    const permissionState = permission.state;
    setIsRetryingRefusal(recorder.permissionDenied);
    recorder.clearPermissionDenied();
    let started = false;
    try {
      started = await recorder.start({
        onMicAnswer(result) {
          trackBrainDump("brain_dump_permission_retried", {
            result,
            permission_state: permissionState,
            via,
          });
        },
        shouldRecord: canRecord,
      });
    } finally {
      isRetryingRef.current = false;
      setIsRetryingRefusal(false);
    }
    if (started) onRecording();
    else await permission.refresh();
  }

  return {
    isBlocked: recorder.permissionDenied || isRetryingRefusal,
    permissionState: permission.state,
    retry,
  };
}
