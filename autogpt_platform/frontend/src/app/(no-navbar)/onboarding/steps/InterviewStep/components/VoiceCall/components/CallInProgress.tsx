"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { HeldCall } from "../helpers";
import { CallAura } from "./CallAura";

interface Props {
  held: HeldCall;
  onRejoin: () => void;
  onEnd: () => void;
}

// A call of the person's is still live: this tab rejoins it with the ticket
// it kept, and any tab can end it so the person can start again.
export function CallInProgress({ held, onRejoin, onEnd }: Props) {
  return (
    <div className="flex w-full max-w-md flex-col items-center gap-4">
      <CallAura levels={null} isActive={false} />
      <Text variant="body" align="center">
        {heldCopy(held)}
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-3">
        {held.canRejoin && (
          <Button variant="primary" size="large" onClick={onRejoin}>
            Rejoin
          </Button>
        )}
        <Button
          variant={held.canRejoin ? "secondary" : "primary"}
          size="large"
          onClick={onEnd}
        >
          End that call
        </Button>
      </div>
    </div>
  );
}

function heldCopy(held: HeldCall) {
  if (held.wentElsewhere) {
    return "Your call with Otto went on in another tab or on another device.";
  }
  return held.canRejoin
    ? "Your call with Otto is still going."
    : "Your call with Otto is still going in another tab or on another device.";
}
