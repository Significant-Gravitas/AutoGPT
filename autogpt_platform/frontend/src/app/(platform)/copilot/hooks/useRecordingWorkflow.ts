"use client";

import { postExperimentalReviewSessionRecording } from "@/app/api/__generated__/endpoints/copilot/copilot";
import type { RecordingStartRequestInterpretationRoute } from "@/app/api/__generated__/models/recordingStartRequestInterpretationRoute";
import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import {
  type CapturedStep,
  selectRecordingSettings,
  toCapturedStep,
} from "./recording-helpers";
import type { LocalPCExecutorStatus } from "./useLocalPCExecutor";
import { useRecordingRequests } from "./useRecordingRequests";

export type RecordingPhase =
  | "idle"
  | "recording"
  | "review"
  | "consent"
  | "ready";

export function useRecordingWorkflow(
  sessionID: string | null,
  executor: LocalPCExecutorStatus | undefined,
) {
  const settings = selectRecordingSettings(
    executor?.recording_routes,
    executor?.recording_channels,
  );
  const canStart =
    !!sessionID &&
    executor?.kind === "shim" &&
    (executor.capabilities ?? []).includes("recording") &&
    !!settings;

  const [phase, setPhase] = useState<RecordingPhase>("idle");
  const [steps, setSteps] = useState<CapturedStep[]>([]);
  const [originalStepSeqs, setOriginalStepSeqs] = useState<number[]>([]);
  const [recordingID, setRecordingID] = useState<string | null>(null);
  const [interpretationRoute, setInterpretationRoute] =
    useState<RecordingStartRequestInterpretationRoute>("extract_then_cloud");

  const recordingRequests = useRecordingRequests({
    onStarted: (startedRecording) => {
      setRecordingID(startedRecording.recordingID);
      setInterpretationRoute(startedRecording.interpretationRoute);
      setOriginalStepSeqs([]);
      setSteps([]);
      setPhase("recording");
    },
    onStopped: (recording) => {
      const capturedSteps = (recording.steps ?? []).map(toCapturedStep);
      setSteps(capturedSteps);
      setOriginalStepSeqs(capturedSteps.map((step) => step.seq));
      if (
        recording.interpretation_route === "extract_then_cloud" ||
        recording.interpretation_route === "local_vlm" ||
        recording.interpretation_route === "screenshots_to_cloud"
      ) {
        setInterpretationRoute(recording.interpretation_route);
      }
      setPhase("review");
    },
  });

  const reviewMutation = useMutation({
    mutationFn: ({
      reviewedSessionID,
      reviewedRecordingID,
      removedStepSeqs,
      redactedStepSeqs,
    }: {
      reviewedSessionID: string;
      reviewedRecordingID: string;
      removedStepSeqs: number[];
      redactedStepSeqs: number[];
    }) =>
      postExperimentalReviewSessionRecording(
        reviewedSessionID,
        reviewedRecordingID,
        {
          removed_step_seqs: removedStepSeqs,
          redacted_step_seqs: redactedStepSeqs,
        },
      ).then((response) => {
        if (response.status !== 200) {
          throw new Error("The Local PC executor did not apply the review");
        }
        return response.data;
      }),
    onSuccess: () => setPhase("ready"),
  });

  function start() {
    if (!sessionID || !settings) return;
    reviewMutation.reset();
    recordingRequests.start(sessionID, settings);
  }

  function deleteStep(seq: number) {
    setSteps((previous) => previous.filter((step) => step.seq !== seq));
  }

  function redactStep(seq: number) {
    setSteps((previous) =>
      previous.map((step) =>
        step.seq === seq ? { ...step, redacted: true, value: null } : step,
      ),
    );
  }

  function submitReview() {
    if (!sessionID || !recordingID) return;
    const retainedStepSeqs = new Set(steps.map((step) => step.seq));
    reviewMutation.mutate({
      reviewedSessionID: sessionID,
      reviewedRecordingID: recordingID,
      removedStepSeqs: originalStepSeqs.filter(
        (seq) => !retainedStepSeqs.has(seq),
      ),
      redactedStepSeqs: steps
        .filter((step) => step.redacted)
        .map((step) => step.seq),
    });
  }

  function approve() {
    if (interpretationRoute === "screenshots_to_cloud") {
      setPhase("consent");
      return;
    }
    submitReview();
  }

  function onConsentKeepLocal() {
    reviewMutation.reset();
    setPhase("review");
  }

  function reset() {
    recordingRequests.stopActive();
    recordingRequests.reset();
    reviewMutation.reset();
    setOriginalStepSeqs([]);
    setRecordingID(null);
    setSteps([]);
    setPhase("idle");
  }

  const errorMessage = recordingRequests.hasStartError
    ? "Could not start the recording. Check that your local executor is connected and try again."
    : recordingRequests.hasStopError
      ? "Could not stop the recording. It may still be active on your machine; try again."
      : reviewMutation.isError
        ? "Could not apply your review. The recording has not been marked ready."
        : null;

  return {
    canStart,
    shouldRender: canStart || recordingRequests.isStarting || phase !== "idle",
    phase,
    steps,
    errorMessage,
    isStarting: recordingRequests.isStarting,
    isStopping: recordingRequests.isStopping,
    isSubmittingReview: reviewMutation.isPending,
    start,
    stop: recordingRequests.stop,
    deleteStep,
    redactStep,
    approve,
    onConsentSend: submitReview,
    onConsentKeepLocal,
    reset,
  };
}
