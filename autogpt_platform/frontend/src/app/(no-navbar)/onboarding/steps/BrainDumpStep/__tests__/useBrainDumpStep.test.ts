import { act, renderHook, waitFor } from "@testing-library/react";
import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  onTestFinished,
  vi,
} from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { RecordingPart } from "../recordingStore";

// A snapshot per render, exactly like the real hook: mutating
// `recorderState` after a render does NOT reach the handlers already
// closed over — which is why the hook passes ids and durations around by
// argument instead of reading them off the recorder.
const { captureException, recorderState } = vi.hoisted(() => ({
  captureException: vi.fn(),
  recorderState: {
    phase: "idle" as "idle" | "recording" | "stopping" | "stopped",
    hitTimeLimit: false,
    elapsedSeconds: 0,
    isOffline: false,
    isSavedLocally: false,
    audioStream: null as MediaStream | null,
    hasSpoken: false,
    permissionDenied: false,
    micUnavailable: false,
    recordingId: null as string | null,
    mimeType: "audio/webm",
    clearPermissionDenied: vi.fn(),
    start: vi.fn(),
    stop: vi.fn(),
    flushUploads: vi.fn(),
    resetQueue: vi.fn(),
    findRecoverable: vi.fn(),
    adoptRecovered: vi.fn(),
    resendAllParts: vi.fn(),
    getElapsedSeconds: vi.fn(),
  },
}));

vi.mock("../useBrainDumpRecorder", () => ({
  useBrainDumpRecorder: () => ({ ...recorderState }),
}));

const finalizeBrainDump = vi.fn();
const discardBrainDump = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/brain-dump/brain-dump", () => ({
  finalizeBrainDump: (...args: unknown[]) => finalizeBrainDump(...args),
  discardBrainDump: (...args: unknown[]) => discardBrainDump(...args),
}));

const clearRecording = vi.fn();
const getMetaById = vi.fn();
const getParts = vi.fn();
const saveMeta = vi.fn();
vi.mock("../recordingStore", () => ({
  clearRecording: (...args: unknown[]) => clearRecording(...args),
  getMetaById: (...args: unknown[]) => getMetaById(...args),
  getParts: (...args: unknown[]) => getParts(...args),
  saveMeta: (...args: unknown[]) => saveMeta(...args),
}));

const trackBrainDump = vi.fn();
vi.mock("@/services/onboarding/brain-dump-analytics", () => ({
  trackBrainDump: (...args: unknown[]) => trackBrainDump(...args),
}));

vi.mock("@sentry/nextjs", () => ({ captureException }));

// A wide screen: the live prompt is hidden at rest, beside the ambient cards.
vi.mock("../useIsWideLayout", () => ({ useIsWideLayout: () => true }));

import { useOnboardingWizardStore } from "../../../store";
import { useBrainDumpStep } from "../useBrainDumpStep";

const INTRO_PATH_KEY = "autogpt:onboarding-intro-path";

function completed() {
  return { status: 200, data: { status: "completed" } };
}

function failedWith(errorCode: string) {
  return Promise.resolve({
    status: 200,
    data: { status: "failed", error_code: errorCode },
  });
}

function part(index: number, recordingId = "rec-1"): RecordingPart {
  return {
    id: `${recordingId}:${index}`,
    recordingId,
    partIndex: index,
    blob: new Blob(["chunk"], { type: "audio/webm" }),
    savedAt: 1,
    uploaded: false,
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

async function renderStep() {
  const view = renderHook(() => useBrainDumpStep());
  // Let the mount-time recovery check settle so every test starts from a
  // screen the hook has finished deciding on.
  await act(async () => {
    await Promise.resolve();
    await Promise.resolve();
  });
  return view;
}

function events() {
  return trackBrainDump.mock.calls.map(([name]) => name);
}

function finalizeBody(callIndex = 0) {
  return finalizeBrainDump.mock.calls[callIndex]?.[0];
}

beforeEach(() => {
  vi.clearAllMocks();
  Object.assign(recorderState, {
    phase: "idle",
    hitTimeLimit: false,
    elapsedSeconds: 0,
    isOffline: false,
    isSavedLocally: false,
    audioStream: null,
    hasSpoken: false,
    permissionDenied: false,
    micUnavailable: false,
    recordingId: null,
    mimeType: "audio/webm",
  });
  recorderState.start.mockResolvedValue(true);
  recorderState.clearPermissionDenied.mockImplementation(() => {
    recorderState.permissionDenied = false;
  });
  recorderState.stop.mockResolvedValue(0);
  recorderState.flushUploads.mockResolvedValue(true);
  recorderState.findRecoverable.mockResolvedValue(null);
  recorderState.adoptRecovered.mockResolvedValue(undefined);
  recorderState.resendAllParts.mockResolvedValue([]);
  recorderState.getElapsedSeconds.mockReturnValue(0);
  finalizeBrainDump.mockResolvedValue(completed());
  discardBrainDump.mockResolvedValue({ status: 200 });
  clearRecording.mockResolvedValue(undefined);
  getMetaById.mockResolvedValue(null);
  getParts.mockResolvedValue([]);
  saveMeta.mockResolvedValue(undefined);
  window.sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().setStepBusy(false);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("useBrainDumpStep — headline", () => {
  it("asks about the user's work", async () => {
    const { result } = await renderStep();

    expect(result.current.headline).toBe("Talk to me about your work");
  });
});

describe("useBrainDumpStep — recording", () => {
  it("moves to the recording screen only when the mic actually opened", async () => {
    const { result } = await renderStep();
    expect(result.current.screen).toBe("rest");

    recorderState.start.mockResolvedValue(false);
    await act(async () => {
      await result.current.handleStart();
    });
    expect(result.current.screen).toBe("rest");

    recorderState.start.mockResolvedValue(true);
    await act(async () => {
      await result.current.handleStart();
    });
    expect(result.current.screen).toBe("recording");
  });

  it("tracks cancel and discards the active take", async () => {
    recorderState.recordingId = "rec-1";
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleStop();
    });

    expect(events()).toContain("brain_dump_canceled");
    expect(clearRecording).toHaveBeenCalledWith("rec-1");
    expect(discardBrainDump).toHaveBeenCalledWith({ recording_id: "rec-1" });
    expect(result.current.screen).toBe("rest");
  });

  it("returns to rest and discards the take when stopping fails", async () => {
    recorderState.recordingId = "rec-1";
    recorderState.stop.mockRejectedValueOnce(new Error("stop failed"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleStop();
    });

    expect(result.current.screen).toBe("rest");
    expect(discardBrainDump).toHaveBeenCalledWith({ recording_id: "rec-1" });
    expect(captureException).toHaveBeenCalled();
  });

  it("does not submit a take while cancellation owns it", async () => {
    const stopping = deferred<number>();
    recorderState.recordingId = "rec-1";
    recorderState.stop.mockReturnValue(stopping.promise);
    const { result, rerender } = await renderStep();

    await act(async () => {
      await result.current.handleStart();
    });

    let cancelPromise!: Promise<void>;
    act(() => {
      cancelPromise = result.current.handleStop();
    });
    act(() => {
      recorderState.hitTimeLimit = true;
      rerender();
    });
    await act(async () => {
      await Promise.resolve();
    });

    expect(finalizeBrainDump).not.toHaveBeenCalled();
    expect(result.current.screen).toBe("rest");

    await act(async () => {
      stopping.resolve(30 * 60);
      await cancelPromise;
    });

    expect(discardBrainDump).toHaveBeenCalledWith({ recording_id: "rec-1" });
    expect(result.current.screen).toBe("rest");
  });

  it("does not discard a take while time-limit submission owns it", async () => {
    const finalizing = deferred<ReturnType<typeof completed>>();
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockReturnValue(finalizing.promise);
    const { result, rerender } = await renderStep();

    await act(async () => {
      await result.current.handleStart();
    });
    act(() => {
      recorderState.hitTimeLimit = true;
      rerender();
    });
    await waitFor(() => expect(finalizeBrainDump).toHaveBeenCalled());

    await act(async () => {
      await result.current.handleStop();
    });

    expect(discardBrainDump).not.toHaveBeenCalled();
    expect(recorderState.stop).not.toHaveBeenCalled();

    await act(async () => {
      finalizing.resolve(completed());
      await Promise.resolve();
    });
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
  });

  // The nudge is for someone who has not started talking yet, so it keys
  // on whether the mic has heard anything — not on elapsed time alone.
  it("nudges only while recording, past the threshold, and still silent", async () => {
    recorderState.elapsedSeconds = 6;
    const { result, rerender } = await renderStep();

    expect(result.current.showSilenceNudge).toBe(false);

    await act(async () => {
      await result.current.handleStart();
    });
    expect(result.current.showSilenceNudge).toBe(true);

    recorderState.hasSpoken = true;
    rerender();
    expect(result.current.showSilenceNudge).toBe(false);
  });

  it("keeps the nudge away before the threshold", async () => {
    recorderState.elapsedSeconds = 4;
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleStart();
    });

    expect(result.current.showSilenceNudge).toBe(false);
  });

  it("blocks wizard navigation while the dump is being processed", async () => {
    const inFlight = deferred<ReturnType<typeof completed>>();
    finalizeBrainDump.mockReturnValue(inFlight.promise);
    recorderState.recordingId = "rec-1";
    const { result, unmount } = await renderStep();

    await act(async () => {
      void result.current.handleDone();
      await Promise.resolve();
    });

    expect(result.current.screen).toBe("processing");
    expect(useOnboardingWizardStore.getState().isStepBusy).toBe(true);

    await act(async () => {
      inFlight.resolve(completed());
      await Promise.resolve();
    });
    unmount();

    // Leaving the step must never strand the wizard with Back hidden.
    expect(useOnboardingWizardStore.getState().isStepBusy).toBe(false);
  });
});

describe("useBrainDumpStep — the 30-minute cap", () => {
  // The recorder enforces the cap from the inside. Nothing observed
  // `phase`, so the screen stayed on "recording" with a frozen timer and
  // a closed mic while the user kept talking.
  it("submits the take when the recorder stops itself", async () => {
    recorderState.recordingId = "rec-1";
    recorderState.getElapsedSeconds.mockReturnValue(1800.4);
    const { result, rerender } = await renderStep();

    await act(async () => {
      await result.current.handleStart();
    });
    expect(result.current.screen).toBe("recording");

    recorderState.hitTimeLimit = true;
    await act(async () => {
      rerender();
    });

    await waitFor(() => expect(result.current.reachedTimeLimit).toBe(true));
    expect(finalizeBody()).toMatchObject({
      recording_id: "rec-1",
      duration_secs: 1800.4,
    });
    expect(result.current.screen).not.toBe("recording");
  });

  // A restart stops the recorder too, and the screen stays on "recording"
  // right through it — submitting there would upload the take the user
  // just threw away.
  it("stays quiet while a restart stops the recorder", async () => {
    recorderState.recordingId = "rec-old";
    const stopped = deferred<number>();
    recorderState.stop.mockReturnValue(stopped.promise);
    const { result, rerender } = await renderStep();

    await act(async () => {
      await result.current.handleStart();
    });

    let restarting: Promise<void> | undefined;
    await act(async () => {
      restarting = result.current.handleRestart();
      await Promise.resolve();
    });

    // Mid-restart the recorder is stopped and the screen still says
    // "recording" — the same shape as the hard stop, but this take is
    // being thrown away, not submitted.
    recorderState.phase = "stopped";
    await act(async () => {
      rerender();
      stopped.resolve(0);
      await restarting;
    });

    expect(finalizeBrainDump).not.toHaveBeenCalled();
    expect(result.current.screen).toBe("recording");
  });
});

describe("useBrainDumpStep — finishing a take", () => {
  it("shows the failure state when stopping the recorder fails", async () => {
    recorderState.recordingId = "rec-1";
    recorderState.stop.mockRejectedValueOnce(new Error("stop failed"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
    expect(finalizeBrainDump).not.toHaveBeenCalled();
    expect(captureException).toHaveBeenCalled();
  });

  it("finalizes with the duration reported by stop(), not the last render", async () => {
    // The render's value is stale by however long stopping took, and the
    // backend splits long recordings on this number.
    recorderState.elapsedSeconds = 12;
    recorderState.recordingId = "rec-1";
    recorderState.stop.mockResolvedValue(64.5);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(finalizeBody()).toEqual({
      recording_id: "rec-1",
      input_mode: "voice",
      duration_secs: 64.5,
      mime_type: "audio/webm",
    });
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_completed", {
      duration_secs: 65,
      input_mode: "voice",
      finalize_latency_ms: expect.any(Number),
      // No role was picked, and the take never reached a screen with a prompt.
      prompt_set: "Other",
      prompts_seen: 0,
    });
    // The finalize round trip rides on the outcome, not on an event of its own.
    expect(trackBrainDump).not.toHaveBeenCalledWith(
      "finalize_latency_ms",
      expect.anything(),
    );
  });

  it("marks the take finalized by id and clears it before advancing", async () => {
    recorderState.recordingId = "rec-1";
    getMetaById.mockResolvedValue({
      recordingId: "rec-1",
      mimeType: "audio/webm",
      startedAt: 5,
      durationSecs: 30,
      finalized: false,
    });
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    // By id, not "the newest take" — a second tab may have started one.
    expect(getMetaById).toHaveBeenCalledWith("rec-1");
    expect(saveMeta).toHaveBeenCalledWith(
      expect.objectContaining({ recordingId: "rec-1", finalized: true }),
    );
    expect(clearRecording).toHaveBeenCalledWith("rec-1");
    expect(recorderState.resetQueue).toHaveBeenCalled();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  // The dump is already safe on the server by this point, so a storage
  // error must not strand the user on the loading screen.
  it("advances even when local cleanup fails", async () => {
    recorderState.recordingId = "rec-1";
    getMetaById.mockRejectedValue(new Error("no indexeddb"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("fails without calling finalize when parts are still unsent", async () => {
    recorderState.recordingId = "rec-1";
    recorderState.flushUploads.mockResolvedValue(false);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
    expect(finalizeBrainDump).not.toHaveBeenCalled();
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });

  it("fails without touching the network when there is no recording", async () => {
    recorderState.recordingId = null;
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
    expect(recorderState.flushUploads).not.toHaveBeenCalled();
    expect(finalizeBrainDump).not.toHaveBeenCalled();
  });

  // A 2xx envelope can still carry a failed pipeline. Advancing on it
  // would hand the user a copilot home built from nothing.
  it("treats a 200 carrying status 'failed' as a failure", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValue({
      status: 200,
      data: { status: "failed", error_code: "transcription_error" },
    });
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
    expect(trackBrainDump).toHaveBeenCalledWith("transcription_failed", {
      error_code: "transcription_error",
      finalize_latency_ms: expect.any(Number),
    });
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
  });

  it("reports the HTTP status as the error code on a non-200", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValue({ status: 500, data: {} });
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
    expect(trackBrainDump).toHaveBeenCalledWith("transcription_failed", {
      error_code: 500,
      finalize_latency_ms: expect.any(Number),
    });
  });

  it("fails rather than throwing when finalize rejects", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockRejectedValue(new Error("offline"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });
});

describe("useBrainDumpStep — insufficient content", () => {
  function insufficient(code = "insufficient_content") {
    return { status: 200, data: { status: "failed", error_code: code } };
  }

  it("shows the recovery screen instead of the failure one", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValueOnce(insufficient("no_usable_speech"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("insufficient");
    expect(trackBrainDump).toHaveBeenCalledWith("transcription_failed", {
      error_code: "no_usable_speech",
      finalize_latency_ms: expect.any(Number),
    });
    // No accidental advance, and nothing personalized to advance to.
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBeNull();
    // The local parts are the user's backup until they pick a next move.
    expect(clearRecording).not.toHaveBeenCalled();
    expect(discardBrainDump).not.toHaveBeenCalled();
  });

  it("keeps a normal transcription failure on the failure screen", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValueOnce({
      status: 200,
      data: { status: "failed", error_code: "transcription_failed" },
    });
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDone();
    });

    expect(result.current.screen).toBe("failed");
  });

  it.each([
    {
      mode: "voice",
      outcome: () => failedWith("storage_failed"),
      reason: "We couldn't store your recording on our side.",
    },
    {
      mode: "typed",
      outcome: () => failedWith("superseded"),
      reason:
        "Something newer you sent, maybe from another tab, replaced this.",
    },
    {
      mode: "voice",
      outcome: () => Promise.reject(new ApiError("Too many", 429, {})),
      reason: "You've tried a few times already. Wait a bit first.",
    },
    {
      mode: "typed",
      outcome: () => Promise.reject(new TypeError("Failed to fetch")),
      reason: "We couldn't reach the server. Check your connection.",
    },
    {
      mode: "voice",
      outcome: () => failedWith("virus_detected"),
      reason: null,
    },
  ])(
    "says why a $mode take failed: $reason",
    async ({ mode, outcome, reason }) => {
      recorderState.recordingId = "rec-1";
      finalizeBrainDump.mockImplementationOnce(outcome);
      const { result } = await renderStep();

      await act(async () => {
        if (mode === "voice") {
          await result.current.handleDone();
          return;
        }
        result.current.setTypedText("I run a bakery.");
      });
      if (mode === "typed") {
        await act(async () => {
          await result.current.handleSubmitTyped();
        });
      }

      expect(result.current.screen).toBe("failed");
      expect(result.current.failureReason).toBe(reason);
    },
  );

  it.each([
    {
      isOffline: true,
      reason: "We couldn't reach the server. Check your connection.",
    },
    { isOffline: false, reason: "Part of your recording didn't reach us." },
  ])(
    "says an upload failed without blaming the connection unless offline ($isOffline)",
    async ({ isOffline, reason }) => {
      recorderState.recordingId = "rec-1";
      recorderState.isOffline = isOffline;
      recorderState.flushUploads.mockResolvedValue(false);
      const { result } = await renderStep();

      await act(async () => {
        await result.current.handleDone();
      });

      expect(result.current.failureReason).toBe(reason);
      expect(finalizeBrainDump).not.toHaveBeenCalled();
    },
  );

  it("recovers into a fresh take when the user records again", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValueOnce(insufficient());
    const { result } = await renderStep();
    await act(async () => {
      await result.current.handleDone();
    });
    expect(result.current.screen).toBe("insufficient");

    await act(async () => {
      await result.current.handleRestart();
    });

    // The rejected take is discarded only now — after the user chose.
    expect(clearRecording).toHaveBeenCalledWith("rec-1");
    expect(discardBrainDump).toHaveBeenCalledWith({ recording_id: "rec-1" });
    expect(result.current.screen).toBe("recording");
  });

  it("opens the composer when the user types instead", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValueOnce(insufficient());
    const { result } = await renderStep();
    await act(async () => {
      await result.current.handleDone();
    });

    act(() => {
      result.current.showTyping();
    });

    expect(result.current.screen).toBe("typing");
  });

  it("still lets the user skip through to the generic path", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockResolvedValueOnce(insufficient());
    const { result } = await renderStep();
    await act(async () => {
      await result.current.handleDone();
    });

    await act(async () => {
      await result.current.handleSkip();
    });

    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("returns a typed dump to recovery with the text preserved", async () => {
    finalizeBrainDump.mockResolvedValueOnce(insufficient());
    const { result } = await renderStep();
    act(() => {
      result.current.showTyping();
    });
    act(() => {
      result.current.setTypedText("hello hello testing");
    });

    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(result.current.screen).toBe("insufficient");
    expect(result.current.typedText).toBe("hello hello testing");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });
});

describe("useBrainDumpStep — restart", () => {
  it("throws away the old take locally and on the server before listening again", async () => {
    recorderState.recordingId = "rec-old";
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRestart();
    });

    expect(recorderState.stop).toHaveBeenCalled();
    expect(recorderState.resetQueue).toHaveBeenCalled();
    expect(clearRecording).toHaveBeenCalledWith("rec-old");
    // Named explicitly: without an id the server drops whatever the row
    // points at, which in a second tab is another take still filling.
    expect(discardBrainDump).toHaveBeenCalledWith({
      recording_id: "rec-old",
    });
    expect(recorderState.start).toHaveBeenCalled();
    expect(events()).toContain("brain_dump_restarted");
  });

  it("survives a failed discard and still starts the new take", async () => {
    recorderState.recordingId = "rec-old";
    clearRecording.mockRejectedValue(new Error("no indexeddb"));
    discardBrainDump.mockRejectedValue(new Error("offline"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRestart();
    });

    expect(recorderState.start).toHaveBeenCalled();
    expect(result.current.screen).not.toBe("failed");
  });

  it("starts a fresh take when stopping the previous recorder fails", async () => {
    recorderState.recordingId = "rec-old";
    recorderState.stop.mockRejectedValueOnce(new Error("stop failed"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRestart();
    });

    expect(discardBrainDump).toHaveBeenCalledWith({
      recording_id: "rec-old",
    });
    expect(recorderState.start).toHaveBeenCalled();
    expect(result.current.screen).toBe("recording");
    expect(captureException).toHaveBeenCalled();
  });

  it("drops back to rest when the mic does not reopen", async () => {
    recorderState.recordingId = "rec-old";
    recorderState.start.mockResolvedValue(false);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRestart();
    });

    expect(result.current.screen).toBe("rest");
  });

  it("discards nothing when there is no take to throw away", async () => {
    recorderState.recordingId = null;
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRestart();
    });

    expect(clearRecording).not.toHaveBeenCalled();
    expect(discardBrainDump).not.toHaveBeenCalled();
  });
});

describe("useBrainDumpStep — retry", () => {
  it("replays the parts and finalizes on the live duration", async () => {
    recorderState.recordingId = "rec-1";
    // The state value on the closure is whatever the last render saw; the
    // live figure is the one the backend needs.
    recorderState.elapsedSeconds = 3;
    recorderState.getElapsedSeconds.mockReturnValue(212.4);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRetry();
    });

    expect(recorderState.resendAllParts).toHaveBeenCalledWith("rec-1");
    expect(finalizeBody()).toMatchObject({
      recording_id: "rec-1",
      duration_secs: 212.4,
    });
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_retry", {
      attempt: 1,
    });
  });

  it("counts each retry attempt", async () => {
    recorderState.recordingId = "rec-1";
    finalizeBrainDump.mockRejectedValue(new Error("offline"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRetry();
    });
    await act(async () => {
      await result.current.handleRetry();
    });

    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_retry", {
      attempt: 2,
    });
  });

  it("retries the recording, not earlier typed text, when sending it failed", async () => {
    finalizeBrainDump.mockRejectedValueOnce(new Error("offline"));
    const { result } = await renderStep();
    act(() => {
      result.current.setTypedText("notes");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });
    recorderState.recordingId = "rec-1";
    recorderState.stop.mockRejectedValueOnce(new Error("stop failed"));
    await act(async () => {
      await result.current.handleDone();
    });
    finalizeBrainDump.mockClear();

    await act(async () => {
      await result.current.handleRetry();
    });

    expect(recorderState.resendAllParts).toHaveBeenCalledWith("rec-1");
    expect(finalizeBody()).toMatchObject({
      recording_id: "rec-1",
      input_mode: "voice",
    });
  });

  it("does not replay parts when there is no recording to retry", async () => {
    recorderState.recordingId = null;
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleRetry();
    });

    expect(recorderState.resendAllParts).not.toHaveBeenCalled();
    expect(result.current.screen).toBe("failed");
  });
});

describe("useBrainDumpStep — typing", () => {
  it("opens the composer when the browser refused the microphone", async () => {
    recorderState.permissionDenied = true;
    const { result } = await renderStep();

    expect(result.current.screen).toBe("typing");
    expect(result.current.isMicBlocked).toBe(true);
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_typed_fallback", {
      reason: "permission_denied",
      prompt_set: "Other",
    });
  });

  it("opens and closes the composer by choice", async () => {
    const { result } = await renderStep();

    act(() => {
      result.current.showTyping();
    });
    expect(result.current.screen).toBe("typing");
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_typed_fallback", {
      reason: "chose_to_type",
      prompt_set: "Other",
    });

    act(() => {
      result.current.showRecording();
    });
    expect(result.current.screen).toBe("rest");
    expect(result.current.isMicBlocked).toBe(false);
  });

  it("ignores a submit with nothing but whitespace", async () => {
    const { result } = await renderStep();

    act(() => {
      result.current.setTypedText("   ");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(finalizeBrainDump).not.toHaveBeenCalled();
    expect(result.current.screen).toBe("rest");
  });

  it("submits trimmed text and advances down the same path as a recording", async () => {
    const { result } = await renderStep();

    act(() => {
      result.current.setTypedText("  invoices every friday  ");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(finalizeBody()).toEqual({
      recording_id: expect.any(String),
      input_mode: "typed",
      text: "invoices every friday",
    });
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_completed", {
      input_mode: "typed",
      chars: "invoices every friday".length,
      prompt_set: "Other",
      prompts_seen: 0,
    });
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("reuses the recording id when the user typed after recording", async () => {
    recorderState.recordingId = "rec-1";
    const { result } = await renderStep();

    act(() => {
      result.current.setTypedText("notes");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(finalizeBody()).toMatchObject({ recording_id: "rec-1" });
  });

  it("fails on a typed dump the pipeline could not process", async () => {
    finalizeBrainDump.mockResolvedValue({
      status: 200,
      data: { status: "failed" },
    });
    const { result } = await renderStep();

    act(() => {
      result.current.setTypedText("notes");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(result.current.screen).toBe("failed");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });

  it("fails rather than throwing when the typed submit rejects", async () => {
    finalizeBrainDump.mockRejectedValue(new Error("offline"));
    const { result } = await renderStep();

    act(() => {
      result.current.setTypedText("notes");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(result.current.screen).toBe("failed");
  });
});

describe("useBrainDumpStep — prompts", () => {
  it("reports the role's prompt set and the prompts seen with a voice take", async () => {
    useOnboardingWizardStore.getState().setRole("Operations");
    recorderState.recordingId = "rec-1";
    recorderState.stop.mockResolvedValue(42);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleStart();
    });
    act(() => {
      result.current.prompt.advance("click");
    });
    await act(async () => {
      await result.current.handleDone();
    });

    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_completed", {
      duration_secs: 42,
      input_mode: "voice",
      finalize_latency_ms: expect.any(Number),
      prompt_set: "Operations",
      prompts_seen: 2,
    });
  });

  it("reports the prompt set and the prompts seen with a typed dump", async () => {
    useOnboardingWizardStore.getState().setRole("Marketing");
    const { result } = await renderStep();

    act(() => {
      result.current.showTyping();
      result.current.setTypedText("the newsletter never ships");
    });
    await act(async () => {
      await result.current.handleSubmitTyped();
    });

    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_typed_fallback", {
      reason: "chose_to_type",
      prompt_set: "Marketing",
    });
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_completed", {
      input_mode: "typed",
      chars: "the newsletter never ships".length,
      prompt_set: "Marketing",
      prompts_seen: 1,
    });
  });

  it("reports the prompt set and the prompts seen with a skip", async () => {
    useOnboardingWizardStore.getState().setRole("HR/People");
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleSkip();
    });

    // A wide screen never shows the live prompt at rest.
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_skipped", {
      prompt_set: "HR/People",
      prompts_seen: 0,
    });
  });

  it("matches Other by its id and never by the text typed beside it", async () => {
    useOnboardingWizardStore.getState().setRole("Other");
    useOnboardingWizardStore.getState().setOtherRole("Engineering");
    const { result } = await renderStep();

    expect(result.current.prompt.promptSet).toBe("Other");
  });
});

describe("useBrainDumpStep — a refused microphone", () => {
  function answerWith(answer: "granted" | "denied" | "unavailable") {
    recorderState.start.mockImplementation(
      async (options?: { onMicAnswer?: (answer: string) => void }) => {
        options?.onMicAnswer?.(answer);
        if (answer === "denied") recorderState.permissionDenied = true;
        return answer === "granted";
      },
    );
  }

  async function renderRefused() {
    recorderState.permissionDenied = true;
    const view = await renderStep();
    expect(view.result.current.screen).toBe("typing");
    expect(view.result.current.isMicBlocked).toBe(true);
    return view;
  }

  function trackedCount(event: string) {
    return events().filter((name) => name === event).length;
  }

  // No Permissions API unless a test brings one: happy-dom answers
  // "granted" for everything.
  beforeEach(() => {
    vi.stubGlobal("navigator", { ...navigator, permissions: undefined });
  });

  it("starts recording when a retry is allowed", async () => {
    answerWith("granted");
    const { result } = await renderRefused();

    await act(async () => {
      await result.current.handleRetryMicrophone();
    });

    expect(recorderState.clearPermissionDenied).toHaveBeenCalled();
    expect(result.current.screen).toBe("recording");
    expect(result.current.isMicBlocked).toBe(false);
    // No Permissions API in this browser.
    expect(trackBrainDump).toHaveBeenCalledWith(
      "brain_dump_permission_retried",
      {
        result: "granted",
        permission_state: "unknown",
        via: "retry_link",
      },
    );
  });

  it("stays on typing and counts one fallback when refused again", async () => {
    answerWith("denied");
    const { result } = await renderRefused();

    await act(async () => {
      await result.current.handleRecordInstead();
    });

    expect(result.current.screen).toBe("typing");
    expect(result.current.isMicBlocked).toBe(true);
    expect(trackBrainDump).toHaveBeenCalledWith(
      "brain_dump_permission_retried",
      {
        result: "denied",
        permission_state: "unknown",
        via: "record_instead",
      },
    );
    expect(trackedCount("brain_dump_typed_fallback")).toBe(1);
  });

  it("drops the blocked notice when the retry finds no microphone", async () => {
    answerWith("unavailable");
    const { result, rerender } = await renderRefused();

    await act(async () => {
      await result.current.handleRetryMicrophone();
    });
    // The real recorder clears its refusal with a state update; this
    // snapshot double needs the render that update would cause.
    rerender();

    expect(result.current.screen).toBe("typing");
    expect(result.current.isMicBlocked).toBe(false);
    expect(trackBrainDump).toHaveBeenCalledWith(
      "brain_dump_permission_retried",
      expect.objectContaining({ result: "unavailable" }),
    );
  });

  it("asks the browser once per click, however fast the clicks come", async () => {
    let answer!: (started: boolean) => void;
    recorderState.start.mockReturnValue(
      new Promise<boolean>((resolve) => {
        answer = resolve;
      }),
    );
    const { result } = await renderRefused();

    await act(async () => {
      void result.current.handleRetryMicrophone();
      void result.current.handleRetryMicrophone();
      answer(true);
    });

    expect(recorderState.start).toHaveBeenCalledTimes(1);
  });

  it("lets the microphone go when it is allowed after the typed text was sent", async () => {
    let allow!: () => void;
    let started: boolean | undefined;
    recorderState.start.mockImplementation(
      (options?: { shouldRecord?: () => boolean }) =>
        new Promise<boolean>((resolve) => {
          allow = () => {
            started = options?.shouldRecord?.() !== false;
            resolve(started);
          };
        }),
    );
    finalizeBrainDump.mockReturnValue(
      deferred<ReturnType<typeof completed>>().promise,
    );
    const { result } = await renderRefused();

    act(() => {
      void result.current.handleRetryMicrophone();
      result.current.setTypedText("I run a bakery.");
    });
    act(() => {
      void result.current.handleSubmitTyped();
    });
    await waitFor(() => expect(result.current.screen).toBe("processing"));

    await act(async () => {
      allow();
    });

    expect(started).toBe(false);
    expect(result.current.screen).toBe("processing");
  });

  it("keeps the typed text on its way when a retry is refused meanwhile", async () => {
    let refuse!: () => void;
    recorderState.start.mockImplementation(
      () =>
        new Promise<boolean>((resolve) => {
          refuse = () => {
            recorderState.permissionDenied = true;
            resolve(false);
          };
        }),
    );
    finalizeBrainDump.mockReturnValue(
      deferred<ReturnType<typeof completed>>().promise,
    );
    const { result, rerender } = await renderRefused();

    act(() => {
      void result.current.handleRetryMicrophone();
      result.current.setTypedText("I run a bakery.");
    });
    act(() => {
      void result.current.handleSubmitTyped();
    });
    await waitFor(() => expect(result.current.screen).toBe("processing"));

    await act(async () => {
      refuse();
    });
    rerender();

    expect(result.current.screen).toBe("processing");
  });

  it("sends the typed text once however fast it is submitted", async () => {
    finalizeBrainDump.mockReturnValue(
      deferred<ReturnType<typeof completed>>().promise,
    );
    const { result } = await renderRefused();
    act(() => {
      result.current.setTypedText("I run a bakery.");
    });

    act(() => {
      void result.current.handleSubmitTyped();
      void result.current.handleSubmitTyped();
    });

    await waitFor(() => expect(finalizeBrainDump).toHaveBeenCalledTimes(1));
  });

  it("keeps the unavailable notice, not the blocked one, while that retry runs", async () => {
    recorderState.permissionDenied = false;
    recorderState.micUnavailable = true;
    recorderState.start.mockReturnValue(new Promise<boolean>(() => {}));
    const { result } = await renderStep();
    act(() => {
      result.current.showTyping();
    });

    act(() => {
      void result.current.handleRetryMicrophone();
    });

    expect(result.current.isMicBlocked).toBe(false);
    expect(result.current.isMicUnavailable).toBe(true);
  });

  it("goes back to the orb from record instead when nothing was refused", async () => {
    const { result } = await renderStep();
    act(() => {
      result.current.showTyping();
    });

    act(() => {
      result.current.handleRecordInstead();
    });

    expect(result.current.screen).toBe("rest");
    expect(recorderState.start).not.toHaveBeenCalled();
  });

  it("clears the blocked state when the site is allowed", async () => {
    const status = Object.assign(new EventTarget(), {
      state: "denied" as PermissionState,
    });
    vi.stubGlobal("navigator", {
      ...navigator,
      permissions: { query: vi.fn().mockResolvedValue(status) },
    });
    const { result } = await renderRefused();
    await waitFor(() => expect(result.current.micPermission).toBe("denied"));

    act(() => {
      status.state = "granted";
      status.dispatchEvent(new Event("change"));
    });

    expect(result.current.isMicBlocked).toBe(false);
    expect(result.current.screen).toBe("typing");
    expect(trackBrainDump).toHaveBeenCalledWith(
      "brain_dump_permission_retried",
      {
        result: "granted",
        permission_state: "granted",
        via: "permission_change",
      },
    );
  });

  it("clears it without a retry event when the block is only reset", async () => {
    const status = Object.assign(new EventTarget(), {
      state: "denied" as PermissionState,
    });
    vi.stubGlobal("navigator", {
      ...navigator,
      permissions: { query: vi.fn().mockResolvedValue(status) },
    });
    const { result } = await renderRefused();
    await waitFor(() => expect(result.current.micPermission).toBe("denied"));

    act(() => {
      status.state = "prompt";
      status.dispatchEvent(new Event("change"));
    });

    expect(result.current.isMicBlocked).toBe(false);
    expect(trackedCount("brain_dump_permission_retried")).toBe(0);
  });
});

describe("useBrainDumpStep — skip", () => {
  it("records the skip and takes path B", async () => {
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleSkip();
    });

    expect(finalizeBody()).toEqual({
      recording_id: expect.any(String),
      input_mode: "skipped",
    });
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  // Back from Hire offers the take again, so the recording stays in the
  // browser. Only the server's part buffer goes, and the skip is recorded
  // under its own id: under the take's, a later send of it reads as done.
  it("keeps a recorded take it is skipping past", async () => {
    recorderState.recordingId = "rec-1";
    const stored = {
      recordingId: "rec-1",
      mimeType: "audio/webm",
      startedAt: 5,
      durationSecs: 120,
      finalized: false,
      userID: "u1",
    };
    getMetaById.mockResolvedValue(stored);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleSkip();
    });

    // Marked so onboarding's completion clears it, and nothing else does.
    expect(saveMeta).toHaveBeenCalledWith({ ...stored, skipped: true });
    expect(clearRecording).not.toHaveBeenCalled();
    expect(finalizeBody().recording_id).not.toBe("rec-1");
    expect(discardBrainDump).toHaveBeenCalledWith({ recording_id: "rec-1" });
    // The discard closes a half-uploaded take, which would hold the row
    // against the skip if the skip landed first.
    expect(discardBrainDump.mock.invocationCallOrder[0]).toBeLessThan(
      finalizeBrainDump.mock.invocationCallOrder[0],
    );
    expect(recorderState.resetQueue).toHaveBeenCalled();
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
  });

  it("has nothing to throw away when no take was ever recorded", async () => {
    recorderState.recordingId = null;
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleSkip();
    });

    expect(clearRecording).not.toHaveBeenCalled();
    expect(discardBrainDump).not.toHaveBeenCalled();
  });

  // Both `handleSkip` and the finalize in flight call `nextStep()`, and
  // the second one lands past the last step — a blank screen with Back
  // and Log out hidden, escapable only by refreshing.
  it("ignores a skip while a submit is already in flight", async () => {
    const inFlight = deferred<ReturnType<typeof completed>>();
    finalizeBrainDump.mockReturnValue(inFlight.promise);
    recorderState.recordingId = "rec-1";
    const { result } = await renderStep();

    await act(async () => {
      void result.current.handleDone();
      await Promise.resolve();
    });
    expect(result.current.screen).toBe("processing");

    await act(async () => {
      await result.current.handleSkip();
    });
    expect(finalizeBrainDump).toHaveBeenCalledTimes(1);

    await act(async () => {
      inFlight.resolve(completed());
      await Promise.resolve();
    });

    // One advance, down the path the recording earned.
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
  });

  // The wizard moves on before the skip is recorded, so the guard is all
  // that stops a second skip from advancing it again, past Hire.
  it("advances once when a second skip lands before the first is recorded", async () => {
    const recording = deferred<ReturnType<typeof completed>>();
    finalizeBrainDump.mockReturnValue(recording.promise);
    const { result } = await renderStep();

    await act(async () => {
      void result.current.handleSkip();
      void result.current.handleSkip();
      await Promise.resolve();
    });
    await act(async () => {
      recording.resolve(completed());
      await Promise.resolve();
    });

    expect(finalizeBrainDump).toHaveBeenCalledTimes(1);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("offers a skipped take back after Back, ready to download", async () => {
    const createObjectURL = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValue("blob:dump");
    const revokeObjectURL = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined);
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    onTestFinished(() => {
      [createObjectURL, revokeObjectURL, click].forEach((spy) =>
        spy.mockRestore(),
      );
    });
    recorderState.recordingId = "rec-skipped";
    const first = await renderStep();
    await act(async () => {
      await first.result.current.handleSkip();
    });
    first.unmount();

    recorderState.recordingId = null;
    recorderState.mimeType = "audio/webm";
    recorderState.findRecoverable.mockResolvedValue({
      recordingId: "rec-skipped",
      mimeType: "audio/mp4",
      durationSecs: 30,
    });
    getParts.mockResolvedValue([part(0, "rec-skipped")]);
    const back = await renderStep();
    await waitFor(() => expect(back.result.current.screen).toBe("recovery"));

    await act(async () => {
      await back.result.current.handleDownloadRecording();
    });

    expect(getParts).toHaveBeenLastCalledWith("rec-skipped");
    expect(click).toHaveBeenCalledTimes(1);
    expect((createObjectURL.mock.calls[0][0] as Blob).type).toBe("audio/mp4");
  });

  // The take on offer can be one another tab is still recording.
  it("leaves the take on offer alone when skipping from the recovery screen", async () => {
    recorderState.findRecoverable.mockResolvedValue({
      recordingId: "rec-other-tab",
      mimeType: "audio/webm",
      durationSecs: 30,
    });
    getParts.mockResolvedValue([part(0, "rec-other-tab")]);
    const { result } = await renderStep();
    await waitFor(() => expect(result.current.screen).toBe("recovery"));

    await act(async () => {
      await result.current.handleSkip();
    });

    expect(discardBrainDump).not.toHaveBeenCalled();
    expect(saveMeta).not.toHaveBeenCalled();
    expect(finalizeBody()).toMatchObject({ input_mode: "skipped" });
  });

  // Back from Hire is a click away while the skip's discard is still on the
  // wire. Landing after the recovered take is resent, it would drop the
  // take's buffer and close it.
  it("holds a recovered take until the skip before it has been recorded", async () => {
    const discard = deferred<{ status: number }>();
    discardBrainDump.mockReturnValueOnce(discard.promise);
    recorderState.recordingId = "rec-skipped";
    const first = await renderStep();
    await act(async () => {
      void first.result.current.handleSkip();
      await Promise.resolve();
    });
    first.unmount();

    recorderState.recordingId = null;
    recorderState.findRecoverable.mockResolvedValue({
      recordingId: "rec-skipped",
      mimeType: "audio/webm",
      durationSecs: 30,
    });
    getParts.mockResolvedValue([part(0, "rec-skipped")]);
    const back = await renderStep();
    await waitFor(() => expect(back.result.current.screen).toBe("recovery"));
    await act(async () => {
      void back.result.current.handleResumeRecovered();
      await Promise.resolve();
    });
    expect(recorderState.adoptRecovered).not.toHaveBeenCalled();

    await act(async () => {
      discard.resolve({ status: 200 });
      await Promise.resolve();
    });

    await waitFor(() =>
      expect(recorderState.adoptRecovered).toHaveBeenCalledWith(
        "rec-skipped",
        "audio/webm",
      ),
    );
    expect(discardBrainDump.mock.invocationCallOrder[0]).toBeLessThan(
      recorderState.adoptRecovered.mock.invocationCallOrder[0],
    );
  });

  // Being unable to say "no thanks" because the network is down would be
  // absurd — path B is the safe default anyway.
  it("lets the user through even when the skip call fails", async () => {
    finalizeBrainDump.mockRejectedValue(new Error("offline"));
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleSkip();
    });

    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("B");
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });
});

describe("useBrainDumpStep — recovery", () => {
  const recovered = {
    recordingId: "rec-crashed",
    mimeType: "audio/webm",
    durationSecs: 95,
  };

  it("offers an unfinalized take that still has parts on disk", async () => {
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockResolvedValue([
      part(0, "rec-crashed"),
      part(1, "rec-crashed"),
    ]);

    const { result } = await renderStep();

    await waitFor(() => expect(result.current.screen).toBe("recovery"));
    expect(result.current.recoverable).toEqual(recovered);
    expect(trackBrainDump).toHaveBeenCalledWith("brain_dump_recovery_shown", {
      parts: 2,
    });
  });

  // Metadata without parts is a take with no audio in it — offering it
  // back would promise something that cannot be uploaded.
  it("stays silent when the stored take has no parts", async () => {
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockResolvedValue([]);

    const { result } = await renderStep();

    expect(result.current.screen).toBe("rest");
    expect(result.current.recoverable).toBeNull();
    expect(events()).not.toContain("brain_dump_recovery_shown");
  });

  it("stays silent when the parts cannot be read at all", async () => {
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockRejectedValue(new Error("no indexeddb"));

    const { result } = await renderStep();

    expect(result.current.screen).toBe("rest");
    expect(result.current.recoverable).toBeNull();
  });

  // The recorder writes the adopted id to a ref, so a handler reading
  // `recorder.recordingId` would still see this render's null.
  it("resumes with the recovered id and its stored duration", async () => {
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockResolvedValue([part(0, "rec-crashed")]);
    const { result } = await renderStep();
    await waitFor(() => expect(result.current.screen).toBe("recovery"));

    await act(async () => {
      await result.current.handleResumeRecovered();
    });

    expect(recorderState.adoptRecovered).toHaveBeenCalledWith(
      "rec-crashed",
      "audio/webm",
    );
    expect(finalizeBody()).toMatchObject({
      recording_id: "rec-crashed",
      duration_secs: 95,
    });
    expect(window.sessionStorage.getItem(INTRO_PATH_KEY)).toBe("A");
    expect(events()).toContain("brain_dump_recovery_used");
  });

  it("does not leave recovery while its submission owns the take", async () => {
    const finalizing = deferred<ReturnType<typeof completed>>();
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockResolvedValue([part(0, "rec-crashed")]);
    finalizeBrainDump.mockReturnValue(finalizing.promise);
    const { result } = await renderStep();
    await waitFor(() => expect(result.current.screen).toBe("recovery"));

    act(() => {
      void result.current.handleResumeRecovered();
    });
    await waitFor(() => expect(finalizeBrainDump).toHaveBeenCalled());

    await act(async () => {
      await result.current.handleTypeInsteadOfRecovered();
    });

    expect(result.current.screen).toBe("processing");
    expect(result.current.recoverable).toEqual(recovered);

    await act(async () => {
      finalizing.resolve(completed());
      await Promise.resolve();
    });
    await waitFor(() =>
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2),
    );
  });

  it("releases the server buffer when the take is abandoned", async () => {
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockResolvedValue([part(0, "rec-crashed")]);
    const { result } = await renderStep();
    await waitFor(() => expect(result.current.screen).toBe("recovery"));

    await act(async () => {
      await result.current.handleDiscardRecovered();
    });

    expect(clearRecording).toHaveBeenCalledWith("rec-crashed");
    // Otherwise those chunks sit in Redis until their TTL for a recording
    // nobody will ever finalize.
    expect(discardBrainDump).toHaveBeenCalledWith({
      recording_id: "rec-crashed",
    });
    expect(result.current.recoverable).toBeNull();
    expect(result.current.screen).toBe("rest");
  });

  it("abandons the take and opens the composer when the user types instead", async () => {
    recorderState.findRecoverable.mockResolvedValue(recovered);
    getParts.mockResolvedValue([part(0, "rec-crashed")]);
    const { result } = await renderStep();
    await waitFor(() => expect(result.current.screen).toBe("recovery"));

    await act(async () => {
      await result.current.handleTypeInsteadOfRecovered();
    });

    expect(discardBrainDump).toHaveBeenCalledWith({
      recording_id: "rec-crashed",
    });
    expect(result.current.screen).toBe("typing");
    expect(result.current.recoverable).toBeNull();
  });
});

describe("useBrainDumpStep — download", () => {
  it("stitches the stored parts into one file named after the recording", async () => {
    const createObjectURL = vi.fn().mockReturnValue("blob:dump");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    recorderState.recordingId = "rec-1";
    recorderState.mimeType = "audio/mp4";
    getParts.mockResolvedValue([part(0), part(1)]);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDownloadRecording();
    });

    expect(click).toHaveBeenCalledTimes(1);
    const blob = createObjectURL.mock.calls[0][0] as Blob;
    expect(blob.type).toBe("audio/mp4");
    // Both chunks, in order, rather than just the last one.
    expect(blob.size).toBe(part(0).blob.size + part(1).blob.size);
    // Revoked in the same turn so the object URL does not leak.
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:dump");
    expect(events()).toContain("brain_dump_download");
  });

  it("downloads nothing when there is nothing stored", async () => {
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => undefined);
    recorderState.recordingId = "rec-1";
    getParts.mockResolvedValue([]);
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDownloadRecording();
    });

    expect(click).not.toHaveBeenCalled();
  });

  it("downloads nothing when no take was ever started", async () => {
    recorderState.recordingId = null;
    const { result } = await renderStep();

    await act(async () => {
      await result.current.handleDownloadRecording();
    });

    expect(getParts).not.toHaveBeenCalled();
    expect(events()).not.toContain("brain_dump_download");
  });
});
