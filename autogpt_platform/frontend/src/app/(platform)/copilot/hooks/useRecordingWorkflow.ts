"use client";

import { useState } from "react";
import {
  hasRememberedRecordingConsent,
  recordingKind,
} from "../components/LocalPCRecordingConsent/helpers";
import type { ExecutorStatus } from "./useLocalPCExecutor";

/** One captured step shown in the review-before-send view. Mirrors the
 *  floor fields of the backend TrajectoryStep (§1) the user needs to see
 *  to decide whether a step is safe to send. */
export interface CapturedStep {
  seq: number;
  action: string;
  label?: string | null;
  activeApp?: string | null;
  /** Demonstrated value, if any — shown so the user can spot + redact a
   *  value that shouldn't leave the machine. */
  value?: string | null;
  redacted?: boolean;
}

export type RecordingPhase =
  | "idle"
  | "recording"
  | "review" // stopped; awaiting the user's review-before-send approval
  | "consent" // screenshots_to_cloud route needs the §9.1 consent
  | "sent";

export type InterpretationRoute =
  | "extract_then_cloud"
  | "local_vlm"
  | "screenshots_to_cloud";

export interface UseRecordingWorkflowResult {
  available: boolean;
  phase: RecordingPhase;
  steps: CapturedStep[];
  consentKind: string;
  start: () => void;
  stop: (steps: CapturedStep[]) => void;
  deleteStep: (seq: number) => void;
  redactStep: (seq: number) => void;
  /** Approve sending the reviewed recording. If the route is
   *  screenshots_to_cloud and consent isn't remembered, transitions to the
   *  consent phase instead of sending. */
  approve: () => void;
  onConsentSend: () => void;
  onConsentKeepLocal: () => void;
  reset: () => void;
}

/**
 * Client-side state machine for the record → review → (consent) → send
 * flow. The actual capture + interpretation happen on the shim / agent
 * side; this hook owns the *frontend affordance*: when the Record button
 * is available (shim advertises the `recording` capability), the active
 * indicator, the review-before-send buffer, and the §9.1 consent gate for
 * the screenshots_to_cloud route.
 *
 * "Demonstration mode requires approval before the recording leaves the
 * machine" — so the default flow stops into `review`, never auto-sends.
 */
export function useRecordingWorkflow(
  executor: ExecutorStatus | undefined,
  options: { interpretationRoute?: InterpretationRoute } = {},
): UseRecordingWorkflowResult {
  const route: InterpretationRoute =
    options.interpretationRoute ?? "extract_then_cloud";

  const available =
    executor?.kind === "shim" &&
    (executor.capabilities ?? []).includes("recording");

  const [phase, setPhase] = useState<RecordingPhase>("idle");
  const [steps, setSteps] = useState<CapturedStep[]>([]);

  const consentKind = recordingKind({
    interpretationRoute: route,
    app: executor?.platform,
  });

  function start() {
    setSteps([]);
    setPhase("recording");
  }

  function stop(captured: CapturedStep[]) {
    setSteps(captured);
    setPhase("review");
  }

  function deleteStep(seq: number) {
    setSteps((prev) => prev.filter((s) => s.seq !== seq));
  }

  function redactStep(seq: number) {
    setSteps((prev) =>
      prev.map((s) =>
        s.seq === seq ? { ...s, redacted: true, value: null } : s,
      ),
    );
  }

  function approve() {
    // Only the screenshots_to_cloud route crosses the new line (§3.1/§9.1).
    if (
      route === "screenshots_to_cloud" &&
      !hasRememberedRecordingConsent(consentKind)
    ) {
      setPhase("consent");
      return;
    }
    setPhase("sent");
  }

  function onConsentSend() {
    setPhase("sent");
  }

  function onConsentKeepLocal() {
    // Back to review — the recording stays on the machine; the user can
    // re-decide or discard.
    setPhase("review");
  }

  function reset() {
    setSteps([]);
    setPhase("idle");
  }

  return {
    available,
    phase,
    steps,
    consentKind,
    start,
    stop,
    deleteStep,
    redactStep,
    approve,
    onConsentSend,
    onConsentKeepLocal,
    reset,
  };
}
