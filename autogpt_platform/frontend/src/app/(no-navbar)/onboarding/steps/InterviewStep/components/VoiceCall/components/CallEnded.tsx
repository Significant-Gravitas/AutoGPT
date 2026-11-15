"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { finishFailureCopy, type StartFailure } from "../../../helpers";
import { savedBeforeCopy } from "../earlierTake";
import { type CallEnding, endedCopy } from "../helpers";
import { CallInsufficient } from "./CallInsufficient";

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

// What became of a call's words: read from how the call ended first, then
// from the hand-off's answer.
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
      <CallInsufficient
        onCallAgain={onCallAgain}
        onTypeInstead={onTypeInstead}
        onSkip={onSkip}
      />
    );
  }

  // With nothing to hand off, calling again leads.
  const leadsWithCall =
    ending.kind === "missing" ||
    ending.kind === "no_words" ||
    ending.kind === "unsaved";
  const handOffFailed =
    ending.kind === "no_words" ||
    ending.kind === "retry" ||
    ending.kind === "unsaved" ||
    ending.kind === "saved_before";
  // Meet your team leads while some call's words wait for the person.
  const meetsTeam =
    ending.kind === "team" ||
    ending.kind === "lost" ||
    ending.kind === "saved_before";
  const otherWays = (
    <>
      <Button
        variant={leadsWithCall ? "primary" : "secondary"}
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
      {/* Read out when the call ends, without the ways on below it. */}
      <div role="status" className="flex w-full flex-col items-center gap-4">
        {ending.kind === "missing" ? (
          <Text variant="body" tone="muted" align="center">
            We couldn&apos;t find that call.
          </Text>
        ) : (
          <Text variant="body" tone="muted" align="center">
            {endedCopy(endReason, { handOffFailed })}
          </Text>
        )}
        {ending.kind === "lost" && (
          <Text variant="body" align="center">
            What you said so far is saved.
          </Text>
        )}
        {ending.kind === "saved_before" && (
          <Text variant="body" align="center">
            {savedBeforeCopy(ending.latest)}
          </Text>
        )}
        {ending.kind === "retry" && (
          <Text variant="body" align="center">
            What you said is kept, but we couldn&apos;t finish setting up from
            it yet.
          </Text>
        )}
        {ending.kind === "no_words" && (
          <Text variant="body" align="center">
            We didn&apos;t hear you on that call. You can call again or type
            instead.
          </Text>
        )}
        {ending.kind === "unsaved" && (
          <Text variant="body" align="center">
            Something went wrong saving that call. You can call again or type
            instead.
          </Text>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-center gap-3">
        {meetsTeam && (
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
          ending.kind === "saved_before" ||
          ending.kind === "missing" ||
          ending.kind === "no_words" ||
          ending.kind === "unsaved" ||
          (ending.kind === "retry" && ending.leftOpen)) &&
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
