import {
  discardBrainDump,
  finalizeBrainDump,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import { setIntroPath } from "@/services/onboarding/brain-dump-handoff";
import { useEffect, useRef, useState } from "react";
import * as Sentry from "@sentry/nextjs";
import { useOnboardingWizardStore } from "../../store";
import { trackBrainDump } from "@/services/onboarding/brain-dump-analytics";
import {
  BRAIN_DUMP_HEADLINE,
  failureCauseOf,
  failureReason,
  type FailureCause,
  isInsufficientDump,
  isTypedTextTooLong,
  SILENCE_NUDGE_SECONDS,
} from "./helpers";
import {
  clearRecording,
  getMetaById,
  getParts,
  saveMeta,
} from "./recordingStore";
import { promptSetNameForRole } from "./prompts";
import { useBrainDumpRecorder } from "./useBrainDumpRecorder";
import { useMicrophoneRetry } from "./useMicrophoneRetry";
import { usePromptRotation } from "./usePromptRotation";

// A skip whose discard and record are still on the wire. Module scope, so the
// remount that Back from Hire causes waits for it before sending a new take,
// which the late skip would otherwise replace.
let pendingSkip: Promise<void> = Promise.resolve();

export type ScreenState =
  | "rest"
  | "recording"
  | "processing"
  | "typing"
  | "recovery"
  | "failed"
  | "insufficient";

export function useBrainDumpStep() {
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const setStepBusy = useOnboardingWizardStore((s) => s.setStepBusy);
  // Picked on the step before; "Other" is matched as an id, not its free text.
  const role = useOnboardingWizardStore((s) => s.role);
  const recorder = useBrainDumpRecorder({
    promptSet: promptSetNameForRole(role),
  });

  const [screen, setScreen] = useState<ScreenState>("rest");
  const prompt = usePromptRotation({ role, screen });
  // For answers that arrive after the render that asked, like a permission
  // prompt answered once the typed text is already on its way.
  const screenRef = useRef(screen);
  useEffect(() => {
    screenRef.current = screen;
  });
  const microphone = useMicrophoneRetry({
    recorder,
    canRecord: () => screenRef.current === "typing" && !isSubmittingRef.current,
    onRecording: () => setScreen("recording"),
  });
  // A refused microphone is one fallback to typing per visit, however many
  // times the user then tries the microphone again.
  const typedFallbackTrackedRef = useRef(false);
  // Which submission the quality gate last rejected — the recovery
  // screen's copy and primary action adapt to how the user got there.
  const [insufficientMode, setInsufficientMode] = useState<"voice" | "typed">(
    "voice",
  );
  // Which kind of take was last sent: the failure screen retries that one,
  // and a typed take has no recording to resend.
  const [submittedMode, setSubmittedMode] = useState<"voice" | "typed">(
    "voice",
  );
  const [failure, setFailure] = useState<FailureCause>(null);
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
  const [isTypedTooLong, setIsTypedTooLong] = useState(false);
  const retryCountRef = useRef(0);
  // A submit in flight owns the screen: "Skip for now" must not race
  // `completeAndAdvance` to `nextStep()`, or the wizard advances twice and
  // lands past the last step with nothing to render. A skip takes it for
  // good, since it leaves the step.
  const isSubmittingRef = useRef(false);
  const activeTakeActionRef = useRef<{
    action: "submit" | "discard";
    token: symbol;
  } | null>(null);

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
  }, []);

  // Permission denial is never a dead end: the typing composer takes over
  // under the same headline.
  // A retry refused while the typed text is on its way leaves the submission
  // alone: reopening the composer would let it be sent a second time.
  useEffect(() => {
    if (!recorder.permissionDenied) return;
    if (isSubmittingRef.current || screenRef.current === "processing") return;
    setScreen("typing");
    if (typedFallbackTrackedRef.current) return;
    typedFallbackTrackedRef.current = true;
    trackBrainDump("brain_dump_typed_fallback", {
      reason: "permission_denied",
      prompt_set: prompt.promptSet,
    });
  }, [recorder.permissionDenied, prompt.promptSet]);

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
  }, [screen, recorder.hitTimeLimit]);

  async function handleStart() {
    const started = await recorder.start();
    if (started) setScreen("recording");
  }

  async function handleDone() {
    setSubmittedMode("voice");
    setScreen("processing");
    // Duration comes back from `stop()` for the same reason the id is
    // passed in below — `recorder.elapsedSeconds` here is this render's
    // value, and it is short by however long stopping took.
    try {
      const durationSecs = await recorder.stop();
      await submitRecording(recorder.recordingId, durationSecs);
    } catch (error) {
      Sentry.captureException(error, {
        tags: { component: "useBrainDumpStep", action: "send" },
      });
      fail(null);
    }
  }

  async function handleStop() {
    const recordingId = recorder.recordingId;
    const actionToken = recordingId
      ? claimTakeAction(recordingId, "discard")
      : null;
    if (recordingId && !actionToken) return;
    trackBrainDump("brain_dump_canceled");
    try {
      try {
        await recorder.stop();
      } catch (error) {
        Sentry.captureException(error, {
          tags: { component: "useBrainDumpStep", action: "cancel" },
        });
      }
      recorder.resetQueue();
      if (recordingId) await discardTake(recordingId);
    } finally {
      setScreen("rest");
      if (actionToken) releaseTakeAction(actionToken);
    }
  }

  // The id is passed in rather than read off the recorder: `adoptRecovered`
  // writes it to a ref, and this function's closure would still be holding
  // the previous render's `null`.
  async function submitRecording(
    recordingId: string | null,
    durationSecs: number,
  ) {
    setSubmittedMode("voice");
    const actionToken = recordingId
      ? claimTakeAction(recordingId, "submit")
      : null;
    if (recordingId && !actionToken) {
      if (activeTakeActionRef.current?.action === "discard") {
        setScreen("rest");
      }
      return false;
    }
    isSubmittingRef.current = true;
    try {
      await finalizeRecording(recordingId, durationSecs);
      return true;
    } finally {
      isSubmittingRef.current = false;
      if (actionToken) releaseTakeAction(actionToken);
    }
  }

  async function finalizeRecording(
    recordingId: string | null,
    durationSecs: number,
  ) {
    if (!recordingId) {
      fail(null);
      return;
    }
    const allUploaded = await recorder.flushUploads();
    if (!allUploaded) {
      fail(recorder.isOffline ? "network" : "upload_failed");
      return;
    }

    await pendingSkip;
    await saveWizardAnswers();
    const startedAt = performance.now();
    let finalizeLatencyMs: number;
    try {
      const response = await finalizeBrainDump({
        recording_id: recordingId,
        input_mode: "voice",
        duration_secs: durationSecs,
        mime_type: recorder.mimeType,
      });
      finalizeLatencyMs = Math.round(performance.now() - startedAt);
      if (response.status !== 200 || response.data.status === "failed") {
        const errorCode =
          response.status === 200 ? response.data.error_code : response.status;
        const rejected = isInsufficientDump(errorCode);
        trackBrainDump("transcription_failed", {
          error_code: errorCode,
          finalize_latency_ms: finalizeLatencyMs,
        });
        // The recording itself went through — it just didn't carry enough
        // to personalize from, so the recovery screen offers a fresh take
        // instead of a retry of this one. Nothing is cleared until the
        // user picks their next move.
        if (rejected) {
          setInsufficientMode("voice");
          setScreen("insufficient");
        } else fail(errorCode ?? null);
        return;
      }
    } catch (error) {
      fail(failureCauseOf(error));
      return;
    }

    trackBrainDump("brain_dump_completed", {
      duration_secs: Math.round(durationSecs),
      input_mode: "voice",
      finalize_latency_ms: finalizeLatencyMs,
      ...promptProperties(),
    });
    await completeAndAdvance(recordingId, "A");
  }

  // A fresh take: the current recording is stopped and thrown away — the
  // local parts and the server's half-uploaded buffer both — before the
  // orb starts listening again under a new recording id.
  async function handleRestart() {
    const previousId = recorder.recordingId;
    const actionToken = previousId
      ? claimTakeAction(previousId, "discard")
      : null;
    if (previousId && !actionToken) return;
    trackBrainDump("brain_dump_restarted");
    let started = false;
    try {
      try {
        await recorder.stop();
      } catch (error) {
        Sentry.captureException(error, {
          tags: { component: "useBrainDumpStep", action: "restart" },
        });
      }
      recorder.resetQueue();
      if (previousId) await discardTake(previousId);
      started = await recorder.start();
    } finally {
      setScreen(started ? "recording" : "rest");
      if (actionToken) releaseTakeAction(actionToken);
    }
  }

  async function handleRetry() {
    retryCountRef.current += 1;
    trackBrainDump("brain_dump_retry", { attempt: retryCountRef.current });
    if (submittedMode === "typed") {
      await handleSubmitTyped();
      return;
    }
    setScreen("processing");
    const recordingId = recorder.recordingId;
    if (recordingId) await recorder.resendAllParts(recordingId);
    // Same reason `handleDone` takes the duration from `stop()`: the
    // state value on this closure is whatever the last render saw.
    await submitRecording(recordingId, recorder.getElapsedSeconds());
  }

  function handleShowTyping() {
    setScreen("typing");
    trackBrainDump("brain_dump_typed_fallback", {
      reason: "chose_to_type",
      prompt_set: prompt.promptSet,
    });
  }

  function handleShowRecording() {
    setScreen("rest");
  }

  // With the microphone refused, going back to the orb would only refuse
  // again, so "record instead" asks the browser once more from here.
  function handleRecordInstead() {
    if (microphone.isBlocked) void microphone.retry("record_instead");
    else handleShowRecording();
  }

  function promptProperties() {
    return {
      prompt_set: prompt.promptSet,
      prompts_seen: prompt.promptsSeen(),
    };
  }

  function handleTypedTextChange(text: string) {
    setTypedText(text);
    setIsTypedTooLong(false);
  }

  async function handleSubmitTyped() {
    const text = typedText.trim();
    if (!text || isSubmittingRef.current) return;
    isSubmittingRef.current = true;
    setIsTypedTooLong(false);
    try {
      await submitTyped(text);
    } finally {
      isSubmittingRef.current = false;
    }
  }

  async function submitTyped(text: string) {
    setSubmittedMode("typed");
    setScreen("processing");
    const recordingId = recorder.recordingId ?? crypto.randomUUID();
    await pendingSkip;
    await saveWizardAnswers();
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
        const errorCode =
          response.status === 200 ? response.data.error_code : response.status;
        const rejected = isInsufficientDump(errorCode);
        // The typed text stays in the composer, so "Type instead" from
        // the recovery screen reopens it with nothing lost.
        if (rejected) {
          setInsufficientMode("typed");
          setScreen("insufficient");
        } else fail(errorCode ?? null);
        return;
      }
    } catch (error) {
      // Retrying the same text would fail the same way, so the user goes
      // back to it with the limit spelled out.
      const tooLong = isTypedTextTooLong(error);
      setIsTypedTooLong(tooLong);
      if (tooLong) setScreen("typing");
      else fail(failureCauseOf(error));
      return;
    }
    trackBrainDump("brain_dump_completed", {
      input_mode: "typed",
      chars: text.length,
      ...promptProperties(),
    });
    await completeAndAdvance(recordingId, "A");
  }

  async function handleSkip() {
    // A skip that lands while the dump is being submitted, or behind
    // another skip, would advance the wizard a second time, past Hire.
    if (isSubmittingRef.current) return;
    isSubmittingRef.current = true;
    trackBrainDump("brain_dump_skipped", promptProperties());
    const recordedId = recorder.recordingId;
    recorder.resetQueue();
    setIntroPath("B");
    await saveWizardAnswers();
    // Nothing below can change where a skip goes, so the user is not kept
    // waiting on it.
    nextStep();
    const skip = recordSkip(recordedId);
    pendingSkip = skip;
    await skip;
  }

  // The server builds the team from the role in the saved progress, and a
  // change reaches it only 200 ms later unless flushed.
  async function saveWizardAnswers() {
    await useOnboardingWizardStore
      .getState()
      .flushProgress?.()
      .catch(() => undefined);
  }

  async function recordSkip(recordedId: string | null) {
    // The recording stays in the browser, so Back offers it again until
    // onboarding finishes, and the skip is its own take on the server: under
    // the recording's id, a later send of it would read as already done.
    // The discard goes first, because it also closes a half-uploaded take
    // that would otherwise hold the row against the skip.
    if (recordedId) {
      await markSkipped(recordedId);
      await discardBrainDump({ recording_id: recordedId }).catch(
        () => undefined,
      );
    }
    try {
      await finalizeBrainDump({
        recording_id: crypto.randomUUID(),
        input_mode: "skipped",
      });
    } catch {
      // Path B is the safe default anyway.
    }
  }

  async function markSkipped(recordingId: string) {
    try {
      const meta = await getMetaById(recordingId);
      if (meta) await saveMeta({ ...meta, skipped: true });
    } catch {
      // Unmarked, it stays on offer until the user discards it.
    }
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
    // Best-effort: the dump is already safe on the server, so a storage
    // error here must not strand the user on the loading screen.
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
    await pendingSkip;
    await recorder.adoptRecovered(
      recoverable.recordingId,
      recoverable.mimeType,
    );
    await submitRecording(recoverable.recordingId, recoverable.durationSecs);
  }

  async function handleDiscardRecovered() {
    if (await dropRecoverable()) setScreen("rest");
  }

  async function handleTypeInsteadOfRecovered() {
    if (await dropRecoverable()) handleShowTyping();
  }

  // Abandoning a take also releases the server's half-uploaded buffer —
  // otherwise those chunks sit in Redis until their TTL for a recording
  // nobody will ever finalize.
  async function dropRecoverable() {
    if (!recoverable) return false;
    const actionToken = claimTakeAction(recoverable.recordingId, "discard");
    if (!actionToken) return false;
    try {
      await discardTake(recoverable.recordingId);
      setRecoverable(null);
      return true;
    } finally {
      releaseTakeAction(actionToken);
    }
  }

  async function discardTake(recordingId: string) {
    await clearRecording(recordingId).catch(() => undefined);
    await discardBrainDump({ recording_id: recordingId }).catch(
      () => undefined,
    );
  }

  function claimTakeAction(recordingId: string, action: "submit" | "discard") {
    if (activeTakeActionRef.current) return null;
    const token = Symbol(recordingId);
    activeTakeActionRef.current = { action, token };
    return token;
  }

  function releaseTakeAction(token: symbol) {
    if (activeTakeActionRef.current?.token === token) {
      activeTakeActionRef.current = null;
    }
  }

  function fail(cause: FailureCause) {
    setFailure(cause);
    setScreen("failed");
  }

  async function handleDownloadRecording() {
    // Before "Use that recording" adopts it, a kept take is only on offer.
    const take = recorder.recordingId
      ? { recordingId: recorder.recordingId, mimeType: recorder.mimeType }
      : recoverable;
    if (!take) return;
    const { recordingId, mimeType } = take;
    trackBrainDump("brain_dump_download");
    const parts = await getParts(recordingId).catch(() => []);
    if (parts.length === 0) return;
    const blob = new Blob(
      parts.map((part) => part.blob),
      { type: mimeType },
    );
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `brain-dump-${recordingId}.webm`;
    link.click();
    URL.revokeObjectURL(url);
  }

  return {
    headline: BRAIN_DUMP_HEADLINE,
    screen,
    insufficientMode,
    submittedMode,
    failureReason: failureReason(failure),
    typedText,
    setTypedText: handleTypedTextChange,
    elapsedSeconds: recorder.elapsedSeconds,
    isOffline: recorder.isOffline,
    isSavedLocally: recorder.isSavedLocally,
    audioStream: recorder.audioStream,
    showSilenceNudge,
    reachedTimeLimit,
    recoverable,
    handleStart,
    handleStop,
    handleDone,
    handleRestart,
    handleRetry,
    handleSkip,
    handleSubmitTyped,
    isTypedTooLong,
    handleResumeRecovered,
    handleDiscardRecovered,
    handleTypeInsteadOfRecovered,
    handleDownloadRecording,
    showTyping: handleShowTyping,
    showRecording: handleShowRecording,
    handleRecordInstead,
    handleRetryMicrophone: () => microphone.retry("retry_link"),
    isMicBlocked: microphone.isBlocked,
    micPermission: microphone.permissionState,
    isMicUnavailable: recorder.micUnavailable,
    role,
    prompt,
  };
}
