"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { RecordIcon, StopIcon } from "@phosphor-icons/react";
import { useLocalPCExecutor } from "../../hooks/useLocalPCExecutor";
import { useRecordingWorkflow } from "../../hooks/useRecordingWorkflow";
import { LocalPCRecordingConsent } from "../LocalPCRecordingConsent/LocalPCRecordingConsent";
import { RecordingIndicator } from "../RecordingIndicator/RecordingIndicator";
import { RecordingReview } from "../RecordingReview/RecordingReview";

interface Props {
  sessionID: string | null;
}

export function RecordWorkflow({ sessionID }: Props) {
  return (
    <SessionRecordWorkflow
      key={sessionID ?? "no-session"}
      sessionID={sessionID}
    />
  );
}

function SessionRecordWorkflow({ sessionID }: Props) {
  const { data: executor } = useLocalPCExecutor(sessionID);
  const flow = useRecordingWorkflow(sessionID, executor);

  if (!flow.shouldRender) return null;

  return (
    <>
      {flow.phase === "idle" ? (
        <Button
          variant="secondary"
          size="small"
          loading={flow.isStarting}
          disabled={flow.isStarting || !flow.canStart}
          onClick={flow.start}
          aria-label="Record workflow"
        >
          <RecordIcon
            className="h-4 w-4 text-red-600"
            weight="fill"
            aria-hidden="true"
          />
          Record workflow
        </Button>
      ) : null}

      {flow.phase === "recording" ? (
        <div className="flex flex-wrap items-center gap-2">
          <RecordingIndicator stepCount={flow.steps.length} />
          <Button
            variant="secondary"
            size="small"
            loading={flow.isStopping}
            disabled={flow.isStopping}
            onClick={flow.stop}
          >
            <StopIcon className="h-4 w-4" weight="fill" aria-hidden="true" />
            Stop
          </Button>
        </div>
      ) : null}

      {flow.phase === "ready" ? (
        <div className="flex flex-wrap items-center gap-2">
          <Text
            variant="body"
            role="status"
            aria-live="polite"
            className="text-xs text-green-800"
          >
            Recording reviewed. Ask Copilot to generate the skill.
          </Text>
          <Button variant="secondary" size="small" onClick={flow.reset}>
            Record another
          </Button>
        </div>
      ) : null}

      {flow.errorMessage && flow.phase !== "review" ? (
        <Text variant="body" role="alert" className="text-xs text-red-700">
          {flow.errorMessage}
        </Text>
      ) : null}

      <RecordingReview
        isOpen={flow.phase === "review"}
        steps={flow.steps}
        isSubmitting={flow.isSubmittingReview}
        errorMessage={flow.errorMessage}
        onDeleteStep={flow.deleteStep}
        onRedactStep={flow.redactStep}
        onApprove={flow.approve}
        onCancel={flow.reset}
      />

      <LocalPCRecordingConsent
        isOpen={flow.phase === "consent"}
        isSubmitting={flow.isSubmittingReview}
        errorMessage={flow.errorMessage}
        onSendAndBuild={flow.onConsentSend}
        onKeepLocal={flow.onConsentKeepLocal}
      />
    </>
  );
}
