"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { InsufficientState } from "../../../../BrainDumpStep/components/InsufficientState";
import { finishFailureCopy, type StartFailure } from "../../../helpers";
import { type CallEnding, endedCopy } from "../helpers";

interface Props {
  ending: CallEnding;
  endReason: string | null;
  isLeaving: boolean;
  finishFailure: StartFailure | null;
  onMeetTeam: () => void;
  onRetry: () => void;
  onCallAgain: () => void;
  onTypeInstead: () => void;
  onSkip: () => void;
}

// What became of a call's words, read by how the call ended first and the
// hand-off's answer second (research/17 §2.2 step 9).
export function CallEnded({
  ending,
  endReason,
  isLeaving,
  finishFailure,
  onMeetTeam,
  onRetry,
  onCallAgain,
  onTypeInstead,
  onSkip,
}: Props) {
  if (ending.kind === "insufficient") {
    return (
      <InsufficientState
        mode="voice"
        canRecord
        recordLabel="Call again"
        onRecordAgain={onCallAgain}
        onTypeInstead={onTypeInstead}
        onSkip={onSkip}
      />
    );
  }

  const otherWays = (
    <>
      <Button
        variant="secondary"
        size="large"
        disabled={isLeaving}
        onClick={onCallAgain}
      >
        Call again
      </Button>
      <Button
        variant="ghost"
        size="large"
        disabled={isLeaving}
        onClick={onTypeInstead}
      >
        Type instead
      </Button>
    </>
  );

  return (
    <div className="flex w-full max-w-md flex-col items-center gap-4">
      {ending.kind === "missing" ? (
        <Text variant="body" tone="muted" align="center">
          We couldn&apos;t find that call.
        </Text>
      ) : (
        <Text variant="body" tone="muted" align="center">
          {endedCopy(endReason)}
        </Text>
      )}
      {ending.kind === "lost" && (
        <Text variant="body" align="center">
          What you said so far is saved.
        </Text>
      )}
      {ending.kind === "retry" && (
        <Text variant="body" align="center">
          What you said is kept, but we couldn&apos;t finish setting up from it
          yet.
        </Text>
      )}

      <div className="flex flex-wrap items-center justify-center gap-3">
        {(ending.kind === "team" || ending.kind === "lost") && (
          <Button
            variant="primary"
            size="large"
            loading={isLeaving}
            disabled={isLeaving}
            onClick={onMeetTeam}
          >
            Meet your team
          </Button>
        )}
        {ending.kind === "retry" && (
          <Button
            variant="primary"
            size="large"
            loading={isLeaving}
            disabled={isLeaving}
            onClick={onRetry}
          >
            Try again
          </Button>
        )}
        {(ending.kind === "lost" ||
          ending.kind === "missing" ||
          (ending.kind === "retry" && ending.cutOff)) &&
          otherWays}
      </div>

      {finishFailure && (
        <Text variant="small" tone="error" role="alert" align="center">
          {finishFailureCopy(finishFailure)}
        </Text>
      )}
    </div>
  );
}
