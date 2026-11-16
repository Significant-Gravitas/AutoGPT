"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Call02Icon, KeyboardIcon } from "@hugeicons/core-free-icons";
import type { ReactNode } from "react";
import { PrivacyNote } from "../../../../BrainDumpStep/components/PrivacyNote";
import { type CallStartFailure, callStartFailureCopy } from "../helpers";
import { CallAura } from "./CallAura";

interface Props {
  // Testers' pick of how the call ends, when they may pick one.
  endingPicker?: ReactNode;
  failure: CallStartFailure | null;
  onTalk: () => void;
  onTypeInstead: () => void;
}

export function CallChoice({
  endingPicker,
  failure,
  onTalk,
  onTypeInstead,
}: Props) {
  return (
    <div className="flex w-full flex-col items-center gap-5">
      <CallAura levels={null} isActive={false} />
      <Text variant="body" tone="muted" align="center" className="max-w-md">
        Talk it through with Otto on a short call, or type if you&apos;d rather.
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-3">
        <Button
          variant="primary"
          size="large"
          leadingIcon={Call02Icon}
          onClick={onTalk}
        >
          Talk to Otto
        </Button>
        <Button
          variant="secondary"
          size="large"
          leadingIcon={KeyboardIcon}
          onClick={onTypeInstead}
        >
          Type instead
        </Button>
      </div>
      {endingPicker}
      {failure && (
        <Text
          variant="small"
          tone="error"
          role="alert"
          align="center"
          className="max-w-md"
        >
          {callStartFailureCopy(failure)}
        </Text>
      )}
      <PrivacyNote />
    </div>
  );
}
