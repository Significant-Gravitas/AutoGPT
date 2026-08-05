import {
  discardBrainDump,
  finalizeBrainDump,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import { headline, SILENCE_NUDGE_SECONDS } from "./helpers";
import {
  clearRecording,
  getMetaById,
  getParts,
  saveMeta,
} from "./recordingStore";
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
  const [reachedTimeLimit, setReachedTimeLimit] = useState(false);

  // While the recording is being processed there is no way back — the
  // wizard's Back button hides itself on this flag.
  useEffect(() => {
    setStepBusy(screen === "processing");
    return () => setStepBusy(false);
  }, [screen, setStepBusy]);
  const [recoverable, setRecoverable] = useState<{
    recordingId: string;
    mimeType: string;
    durationSecs: number;
  } | null>(null);
  const retryCountRef = useRef(0);
  // A submit in flight owns the screen: "Skip for now" must not race
  // `completeAndAdvance` to `nextStep()`, or the wizard advances twice and
  // lands past the last step with nothing to render.
  const isSubmittingRef = useRef(false);

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

  // Keyed on whether the mic has actually heard anything. It used to key
  // on `!isSavedLocally`, which flips as soon as the first chunk is
  // persisted — MediaRecorder produces one every timeslice whether or not
  // the user spoke, so the nudge could never reach its own threshold.
  const showSilenceNudge =
    screen === "recording" &&
    recorder.elapsedSeconds >= SILENCE_NUDGE_SECONDS &&
    !recorder.hasSpoken;

  // The recorder enforces the 30-minute cap from the inside, so without
  // this the mic would close while the screen still said "recording" — a
  // frozen timer and dead captions until the user thought to press "I'm
  // done". Keyed on the cap specifically, not on the recorder having
  // stopped: a restart stops it too, and that take is being thrown away.
  useEffect(() => {
    if (screen !== "recording" || !recorder.hitTimeLimit) return;
    setReachedTimeLimit(true);
    setScreen("processing");
    void submitRecording(recorder.recordingId, recorder.getElapsedSeconds());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [screen, recorder.hitTimeLimit]);

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
    isSubmittingRef.current = true;
    try {
      await finalizeRecording(recordingId, durationSecs);
    } finally {
      isSubmittingRef.current = false;
    }
  }

  async function finalizeRecording(
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
    const previousId = recorder.recordingId;
    await recorder.stop();
    recorder.resetQueue();
    if (previousId) await clearRecording(previousId).catch(() => undefined);
    // Say which take: without an id the server drops whatever the row
    // currently points at, which in a second tab is somebody else's
    // buffer still being filled.
    if (previousId) {
      await discardBrainDump({ recording_id: previousId }).catch(
        () => undefined,
      );
    }
    const started = await recorder.start();
    if (!started) setScreen("rest");
  }

  async function handleRetry() {
    retryCountRef.current += 1;
    trackBrainDump("brain_dump_retry", { attempt: retryCountRef.current });
    setScreen("processing");
    const recordingId = recorder.recordingId;
    if (recordingId) await recorder.resendAllParts(recordingId);
    // Same reason `handleDone` takes the duration from `stop()`: the
    // state value on this closure is whatever the last render saw.
    await submitRecording(recordingId, recorder.getElapsedSeconds());
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
    isSubmittingRef.current = true;
    try {
      await submitTyped(text);
    } finally {
      isSubmittingRef.current = false;
    }
  }

  async function submitTyped(text: string) {
    setScreen("processing");
    const recordingId = recorder.recordingId ?? crypto.randomUUID();
    try {
      const response = await finalizeBrainDump({
        recording_id: recordingId,
        input_mode: "typed",
        text,
      });
      // A 2xx envelope can still carry a failed pipeline, exactly as on
      // the voice path — advancing on it would hand the user a copilot
      // home built from nothing.
      if (response.status !== 200 || response.data.status === "failed") {
        setScreen("failed");
        return;
      }
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
    // A skip that lands while the dump is being submitted would advance
    // the wizard a second time behind `completeAndAdvance`.
    if (isSubmittingRef.current) return;
    trackBrainDump("brain_dump_skipped");
    const recordedId = recorder.recordingId;
    // Best effort: a failed skip-record still has to let the user
    // through — being unable to say "no thanks" would be absurd.
    try {
      await finalizeBrainDump({
        recording_id: recordedId ?? crypto.randomUUID(),
        input_mode: "skipped",
      });
    } catch {
      // Path B is the safe default anyway.
    }
    // Someone who talked for minutes and then skipped leaves multi-MB
    // blobs behind, and an unfinalized meta row makes the next visit
    // offer back a take they explicitly abandoned.
    if (recordedId) {
      await releaseTake(recordedId);
      await discardBrainDump({ recording_id: recordedId }).catch(
        () => undefined,
      );
    }
    recorder.resetQueue();
    setIntroPath("B");
    nextStep();
  }

  // IndexedDB is only cleared once the server has confirmed the dump —
  // until then the browser is the backup of record.
  async function completeAndAdvance(recordingId: string, path: "A" | "B") {
    await releaseTake(recordingId);
    recorder.resetQueue();
    setIntroPath(path);
    nextStep();
  }

  // Marking finalized before clearing is belt and braces: if the delete
  // fails, the flag alone is enough to stop the take being offered back.
  async function releaseTake(recordingId: string) {
    // Best-effort: the dump is either already safe on the server or
    // deliberately abandoned, so a storage error here must not strand
    // the user on the loading screen.
    try {
      // By id, not "the newest take": a second tab may have started a
      // more recent one, and finalizing that instead would leave the
      // take we actually completed looking recoverable forever.
      const meta = await getMetaById(recordingId);
      if (meta) await saveMeta({ ...meta, finalized: true });
      await clearRecording(recordingId);
    } catch {
      // Nothing left to protect.
    }
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
    await discardBrainDump({ recording_id: recoverable.recordingId }).catch(
      () => undefined,
    );
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
    isSavedLocally: recorder.isSavedLocally,
    audioStream: recorder.audioStream,
    showSilenceNudge,
    reachedTimeLimit,
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
