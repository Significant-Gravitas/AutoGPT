import {
  RecordingPresets,
  requestRecordingPermissionsAsync,
  setAudioModeAsync,
  useAudioRecorder,
  type RecordingOptions,
} from "expo-audio";
import { useRef, useState } from "react";
import { AppState } from "react-native";

import { TranscribeError, transcribe } from "@/lib/api/transcribe";
import { useMountEffect } from "@/lib/useMountEffect";

import {
  MAX_DICTATION_MS,
  MIC_DENIED_MESSAGE,
  NOT_CAUGHT_MESSAGE,
  transcribeFailureMessage,
} from "./dictationHelpers";

// AAC in an MPEG-4 file (.m4a) on both platforms; mono at 64 kbps keeps two
// minutes near 1 MB, far under the route's 25 MB cap.
const DICTATION_OPTIONS: RecordingOptions = {
  ...RecordingPresets.HIGH_QUALITY,
  numberOfChannels: 1,
  bitRate: 64000,
  isMeteringEnabled: true,
};

type Phase = "idle" | "recording" | "transcribing";

interface Args {
  /** The field's text, sent as context so names come out spelled the same. */
  getDraft: () => string;
  onText: (text: string) => void;
}

export function useDictation({ getDraft, onText }: Args) {
  const recorder = useAudioRecorder(DICTATION_OPTIONS);
  const [phase, setPhaseState] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  // Async steps read the phase after an await; state there would be stale.
  const phaseRef = useRef<Phase>("idle");
  const isStarting = useRef(false);
  const isMounted = useRef(true);
  const autoStop = useRef<ReturnType<typeof setTimeout> | null>(null);

  function setPhase(next: Phase) {
    phaseRef.current = next;
    setPhaseState(next);
  }

  function clearAutoStop() {
    if (autoStop.current) clearTimeout(autoStop.current);
    autoStop.current = null;
  }

  async function stopRecorder() {
    try {
      await recorder.stop();
    } catch {
      // Already stopped or released.
    }
    await setAudioModeAsync({ allowsRecording: false }).catch(() => undefined);
  }

  async function start() {
    if (phaseRef.current !== "idle" || isStarting.current) return;
    isStarting.current = true;
    setError(null);
    try {
      const permission = await requestRecordingPermissionsAsync();
      if (!isMounted.current) return;
      if (!permission.granted) {
        setError(MIC_DENIED_MESSAGE);
        return;
      }
      await setAudioModeAsync({ allowsRecording: true, playsInSilentMode: true });
      await recorder.prepareToRecordAsync();
      if (!isMounted.current || AppState.currentState === "background") {
        await stopRecorder();
        return;
      }
      recorder.record();
      setPhase("recording");
      autoStop.current = setTimeout(() => void confirm(), MAX_DICTATION_MS);
    } catch {
      setError(NOT_CAUGHT_MESSAGE);
      await stopRecorder();
    } finally {
      isStarting.current = false;
    }
  }

  async function cancel() {
    if (phaseRef.current !== "recording") return;
    clearAutoStop();
    setPhase("idle");
    await stopRecorder();
  }

  async function confirm() {
    if (phaseRef.current !== "recording") return;
    clearAutoStop();
    setPhase("transcribing");
    await stopRecorder();
    const uri = recorder.uri;
    try {
      if (!uri) throw new TranscribeError(null);
      const text = await transcribe(uri, getDraft());
      if (!isMounted.current) return;
      if (text.trim()) onText(text);
      else setError(NOT_CAUGHT_MESSAGE);
    } catch (failure) {
      if (!isMounted.current) return;
      setError(
        failure instanceof TranscribeError
          ? transcribeFailureMessage(failure.status)
          : NOT_CAUGHT_MESSAGE,
      );
    } finally {
      if (isMounted.current) setPhase("idle");
    }
  }

  // Leaving the app or the screen throws the recording away.
  useMountEffect(() => {
    isMounted.current = true;
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "background") void cancel();
    });
    return () => {
      isMounted.current = false;
      subscription.remove();
      clearAutoStop();
      if (phaseRef.current === "recording") {
        phaseRef.current = "idle";
        void stopRecorder();
      }
    };
  });

  return {
    recorder,
    isActive: phase !== "idle",
    isTranscribing: phase === "transcribing",
    error,
    start,
    cancel,
    confirm,
  };
}
