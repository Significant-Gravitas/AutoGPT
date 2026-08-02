import {
  discardBrainDump,
  finalizeBrainDump,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import { headline, SILENCE_NUDGE_SECONDS } from "./helpers";
import { clearRecording, getMeta, getParts, saveMeta } from "./recordingStore";
import { useBrainDumpRecorder } from "./useBrainDumpRecorder";

export type ScreenState =
  | "rest"
  | "recording"
  | "processing"
  | "typing"
  | "recovery"
  | "failed";

export function useBrainDumpStep() {
  const name = useOnboardingWizardStore((s) => s.name);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const setStepBusy = useOnboardingWizardStore((s) => s.setStepBusy);
  const recorder = useBrainDumpRecorder();

  const [screen, setScreen] = useState<ScreenState>("rest");
  const [typedText, setTypedText] = useState("");

  // While the recording is being processed there is no way back — the
  // wizard's Back button hides itself on this flag.
  useEffect(() => {
    setStepBusy(screen === "processing");
    return () => setStepBusy(false);
  }, [screen, setStepBusy]);
  // Bumped on restart so the live captions remount and drop the words from
  // the discarded take — the screen stays on "recording" throughout, so the
  // caption box would otherwise keep the old transcript on screen.
  const [captionsKey, setCaptionsKey] = useState(0);
  const [recoverable, setRecoverable] = useState<{
    recordingId: string;
    mimeType: string;
    durationSecs: number;
  } | null>(null);
  const retryCountRef = useRef(0);

  useEffect(() => {
    async function checkForRecovery() {
      const meta = await recorder.findRecoverable();
      if (!meta) return;
      const parts = await getParts(meta.recordingId).catch(() => []);
      if (parts.length === 0) return;
      setRecoverable(meta);
      setScreen("recovery");
      trackBrainDump("brain_dump_recovery_shown", { parts: parts.length });
    }
    void checkForRecovery();
    // Recovery is a mount-time question only — re-asking mid-recording
    // would fight the user for control of the screen.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Permission denial is never a dead end: the typing composer takes over
  // under the same headline.
  useEffect(() => {
    if (!recorder.permissionDenied) return;
    setScreen("typing");
    trackBrainDump("brain_dump_typed_fallback", {
      reason: "permission_denied",
    });
  }, [recorder.permissionDenied]);

  const showSilenceNudge =
    screen === "recording" &&
    recorder.elapsedSeconds >= SILENCE_NUDGE_SECONDS &&
    !recorder.isSavedLocally;

  async function handleStart() {
    const started = await recorder.start();
    if (started) setScreen("recording");
  }

  async function handleDone() {
    setScreen("processing");
    // Duration comes back from `stop()` for the same reason the id is
    // passed in below — `recorder.elapsedSeconds` here is this render's
    // value, and it is short by however long stopping took.
    const durationSecs = await recorder.stop();
    await submitRecording(recorder.recordingId, durationSecs);
  }

  // The id is passed in rather than read off the recorder: `adoptRecovered`
  // writes it to a ref, and this function's closure would still be holding
  // the previous render's `null`.
  async function submitRecording(
    recordingId: string | null,
    durationSecs: number,
  ) {
    if (!recordingId) {
      setScreen("failed");
      return;
    }
    const allUploaded = await recorder.flushUploads();
    if (!allUploaded) {
      setScreen("failed");
      return;
    }

    const startedAt = performance.now();
    try {
      const response = await finalizeBrainDump({
        recording_id: recordingId,
        input_mode: "voice",
        duration_secs: durationSecs,
        mime_type: recorder.mimeType,
      });
      trackBrainDump("finalize_latency_ms", {
        ms: Math.round(performance.now() - startedAt),
        input_mode: "voice",
      });
      if (response.status !== 200 || response.data.status === "failed") {
        trackBrainDump("transcription_failed", {
          error_code:
            response.status === 200
              ? response.data.error_code
              : response.status,
        });
        setScreen("failed");
        return;
      }
    } catch {
      setScreen("failed");
      return;
    }

    trackBrainDump("brain_dump_completed", {
      duration_secs: Math.round(durationSecs),
      input_mode: "voice",
    });
    await completeAndAdvance(recordingId, "A");
  }

  // A fresh take: the current recording is stopped and thrown away — the
  // local parts and the server's half-uploaded buffer both — before the
  // orb starts listening again under a new recording id.
  async function handleRestart() {
    trackBrainDump("brain_dump_restarted");
    setCaptionsKey((key) => key + 1);
    const previousId = recorder.recordingId;
    await recorder.stop();
    recorder.resetQueue();
    if (previousId) await clearRecording(previousId).catch(() => undefined);
    await discardBrainDump().catch(() => undefined);
    const started = await recorder.start();
    if (!started) setScreen("rest");
  }

  async function handleRetry() {
    retryCountRef.current += 1;
    trackBrainDump("brain_dump_retry", { attempt: retryCountRef.current });
    setScreen("processing");
    const recordingId = recorder.recordingId;
    if (recordingId) await recorder.resendAllParts(recordingId);
    await submitRecording(recordingId, recorder.elapsedSeconds);
  }

  function handleShowTyping() {
    setScreen("typing");
    trackBrainDump("brain_dump_typed_fallback", { reason: "chose_to_type" });
  }

  function handleShowRecording() {
    setScreen("rest");
  }

  async function handleSubmitTyped() {
    const text = typedText.trim();
    if (!text) return;
    setScreen("processing");
    const recordingId = recorder.recordingId ?? crypto.randomUUID();
    try {
      await finalizeBrainDump({
        recording_id: recordingId,
        input_mode: "typed",
        text,
      });
    } catch {
      setScreen("failed");
      return;
    }
    trackBrainDump("brain_dump_completed", {
      input_mode: "typed",
      chars: text.length,
    });
    await completeAndAdvance(recordingId, "A");
  }

  async function handleSkip() {
    trackBrainDump("brain_dump_skipped");
    const recordingId = recorder.recordingId ?? crypto.randomUUID();
    // Best effort: a failed skip-record still has to let the user
    // through — being unable to say "no thanks" would be absurd.
    try {
      await finalizeBrainDump({
        recording_id: recordingId,
        input_mode: "skipped",
      });
    } catch {
      // Path B is the safe default anyway.
    }
    setIntroPath("B");
    nextStep();
  }

  // IndexedDB is only cleared once the server has confirmed the dump —
  // until then the browser is the backup of record.
  async function completeAndAdvance(recordingId: string, path: "A" | "B") {
    // Local cleanup is best-effort: the dump is already safe on the
    // server, so a storage error here must not strand the user on the
    // loading screen.
    try {
      const meta = await getMeta();
      if (meta) await saveMeta({ ...meta, finalized: true });
      await clearRecording(recordingId);
    } catch {
      // Nothing left to protect — the server has the recording.
    }
    recorder.resetQueue();
    setIntroPath(path);
    nextStep();
  }

  async function handleResumeRecovered() {
    if (!recoverable) return;
    trackBrainDump("brain_dump_recovery_used");
    setScreen("processing");
    await recorder.adoptRecovered(
      recoverable.recordingId,
      recoverable.mimeType,
    );
    await submitRecording(recoverable.recordingId, recoverable.durationSecs);
  }

  async function handleDiscardRecovered() {
    await dropRecoverable();
    setScreen("rest");
  }

  async function handleTypeInsteadOfRecovered() {
    await dropRecoverable();
    handleShowTyping();
  }

  // Abandoning a take also releases the server's half-uploaded buffer —
  // otherwise those chunks sit in Redis until their TTL for a recording
  // nobody will ever finalize.
  async function dropRecoverable() {
    if (!recoverable) return;
    await clearRecording(recoverable.recordingId).catch(() => undefined);
    await discardBrainDump().catch(() => undefined);
    setRecoverable(null);
  }

  async function handleDownloadRecording() {
    const recordingId = recorder.recordingId;
    if (!recordingId) return;
    trackBrainDump("brain_dump_download");
    const parts = await getParts(recordingId).catch(() => []);
    if (parts.length === 0) return;
    const blob = new Blob(
      parts.map((part) => part.blob),
      { type: recorder.mimeType },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `brain-dump-${recordingId}.webm`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return {
    headline: headline(name),
    screen,
    typedText,
    setTypedText,
    elapsedSeconds: recorder.elapsedSeconds,
    isOffline: recorder.isOffline,
    audioStream: recorder.audioStream,
    captionsKey,
    showSilenceNudge,
    recoverable,
    handleStart,
    handleDone,
    handleRestart,
    handleRetry,
    handleSkip,
    handleSubmitTyped,
    handleResumeRecovered,
    handleDiscardRecovered,
    handleTypeInsteadOfRecovered,
    handleDownloadRecording,
    showTyping: handleShowTyping,
    showRecording: handleShowRecording,
    // Going back to the orb is a dead end when the browser already refused
    // the microphone, so the way back is offered only when it can work.
    isMicBlocked: recorder.permissionDenied,
  };
}
