"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Record, Stop } from "@phosphor-icons/react";
import { parseAsString, useQueryState } from "nuqs";
import { useLocalPCExecutor } from "../../hooks/useLocalPCExecutor";
import {
  type CapturedStep,
  useRecordingWorkflow,
} from "../../hooks/useRecordingWorkflow";
import { LocalPCRecordingConsent } from "../LocalPCRecordingConsent/LocalPCRecordingConsent";
import { RecordingIndicator } from "../RecordingIndicator/RecordingIndicator";
import { RecordingReview } from "../RecordingReview/RecordingReview";

/**
 * The "Record workflow" affordance for the copilot UI.
 *
 * Renders nothing unless the connected shim advertises the `recording`
 * capability (read via useLocalPCExecutor). When available it shows a
 * Record button; while recording, a Stop button + the distinct recording
 * indicator; on stop, the review-before-send view; and — only for the
 * screenshots_to_cloud route — the §9.1 calibrated consent dialog.
 *
 * The whole component is mounted behind the WORKFLOW_RECORDING flag in
 * CopilotPage, so it never appears unless the feature is enabled.
 */
export function RecordWorkflow() {
  const [sessionId] = useQueryState("sessionId", parseAsString);
  const { data: executor } = useLocalPCExecutor(sessionId);
  const flow = useRecordingWorkflow(executor);

  if (!flow.available) {
    return null;
  }

  // In v1 the platform doesn't yet receive the live step buffer in the
  // browser (the agent drives capture via MCP tools); the review view is
  // populated by the agent's stop summary. Until that wiring lands, stop
  // hands an empty buffer — the review view shows the empty state and the
  // approval gate still works. This keeps the affordance honest about
  // "approval before it leaves the machine."
  function handleStop() {
    const captured: CapturedStep[] = [];
    flow.stop(captured);
  }

  return (
    <>
      {flow.phase === "recording" ? (
        <div className="flex items-center gap-2">
          <RecordingIndicator stepCount={flow.steps.length} />
          <Button variant="secondary" size="small" onClick={handleStop}>
            <Stop className="h-4 w-4" weight="fill" />
            Stop
          </Button>
        </div>
      ) : (
        <Button
          variant="secondary"
          size="small"
          onClick={flow.start}
          aria-label="Record workflow"
        >
          <Record className="h-4 w-4 text-red-600" weight="fill" />
          Record workflow
        </Button>
      )}

      <RecordingReview
        isOpen={flow.phase === "review"}
        steps={flow.steps}
        onDeleteStep={flow.deleteStep}
        onRedactStep={flow.redactStep}
        onApprove={flow.approve}
        onCancel={flow.reset}
      />

      <LocalPCRecordingConsent
        isOpen={flow.phase === "consent"}
        recordingKind={flow.consentKind}
        onSendAndBuild={flow.onConsentSend}
        onKeepLocal={flow.onConsentKeepLocal}
      />
    </>
  );
}
