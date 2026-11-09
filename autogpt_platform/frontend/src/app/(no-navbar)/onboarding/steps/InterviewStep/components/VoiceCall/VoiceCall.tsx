"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CallEnd01Icon, KeyboardIcon } from "@hugeicons/core-free-icons";
import { PrivacyNote } from "../../../BrainDumpStep/components/PrivacyNote";
import { CallAura } from "./components/CallAura";
import { CallCaptions } from "./components/CallCaptions";
import { CallChoice } from "./components/CallChoice";
import { CallEnded } from "./components/CallEnded";
import { CallInProgress } from "./components/CallInProgress";
import { callStatusCopy, turnsLeftCopy } from "./helpers";
import type { VoiceCallState } from "./useVoiceCall";

interface Props {
  call: VoiceCallState;
}

// The interview as a call: Otto, the captions, the time and turns left, and
// how it ended. The relay runs every limit; this screen only shows them.
export function VoiceCall({ call }: Props) {
  if (call.phase === "checking") {
    return (
      <Text variant="body" tone="muted" role="status">
        One moment…
      </Text>
    );
  }
  if (call.phase === "idle") {
    return (
      <CallChoice
        failure={call.startFailure}
        onTalk={call.talk}
        onTypeInstead={() => void call.typeInstead()}
      />
    );
  }
  if (call.phase === "held" && call.held) {
    return (
      <CallInProgress
        held={call.held}
        onRejoin={call.rejoin}
        onEnd={call.endHeldCall}
      />
    );
  }

  const isOnCall =
    call.phase === "starting" ||
    call.phase === "live" ||
    call.phase === "reconnecting";
  const status = callStatusCopy(call);

  return (
    <div className="flex w-full flex-col items-center gap-5">
      <CallAura levels={call.levels} isActive={call.phase === "live"} />

      {status && (
        <Text variant="body" tone="muted" role="status" align="center">
          {status}
        </Text>
      )}

      {isOnCall && call.timeLeft && call.turnsLeft !== null && (
        <div className="flex items-center gap-4">
          <Text variant="small" tone="muted" tabular>
            {call.timeLeft}
          </Text>
          <Text variant="small" tone="muted">
            {turnsLeftCopy(call.turnsLeft)}
          </Text>
        </div>
      )}

      <CallCaptions lines={call.captions} />

      {call.phase === "ended" && call.ending && (
        <CallEnded
          ending={call.ending}
          endReason={call.endReason}
          isLeaving={call.isLeaving}
          finishFailure={call.finishFailure}
          onMeetTeam={() => void call.meetTeam()}
          onRetry={() => void call.retryFinish()}
          onCallAgain={call.callAgain}
          onTypeInstead={() => void call.typeInstead()}
          onSkip={() => void call.skip()}
        />
      )}

      {isOnCall && (
        <div className="flex flex-wrap items-center justify-center gap-3">
          <Button
            variant="destructive"
            size="large"
            leadingIcon={CallEnd01Icon}
            disabled={call.phase !== "live" || call.isHangingUp}
            onClick={call.hangUp}
          >
            Hang up
          </Button>
          {/* Not while a dropped call re-attaches: its words are still on
              their way, and the relay holds the call for the re-attach. */}
          {call.phase !== "reconnecting" && (
            <Button
              variant="ghost"
              size="large"
              leadingIcon={KeyboardIcon}
              onClick={() => void call.typeInstead()}
            >
              Type instead
            </Button>
          )}
        </div>
      )}

      {(call.phase === "wrapping_up" ||
        call.phase === "saving" ||
        call.phase === "finishing") &&
        call.isPastWindow && (
          <Button
            variant="secondary"
            size="large"
            leadingIcon={KeyboardIcon}
            onClick={() => void call.typeInstead()}
          >
            Type instead
          </Button>
        )}

      {isOnCall && <PrivacyNote />}
    </div>
  );
}
