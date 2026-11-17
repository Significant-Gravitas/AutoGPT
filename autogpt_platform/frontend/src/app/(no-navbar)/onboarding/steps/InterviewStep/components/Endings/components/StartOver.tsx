"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useState } from "react";

interface Props {
  // A call is on offer to the person.
  canCall: boolean;
  // Something the person said is kept, which a new interview replaces.
  hasKept: boolean;
  isDisabled: boolean;
  onOpen: () => void;
  onCallAgain: () => void;
  onTypeInstead: () => void;
}

// A quiet way to do the interview over, under its ending.
export function StartOver({
  canCall,
  hasKept,
  isDisabled,
  onOpen,
  onCallAgain,
  onTypeInstead,
}: Props) {
  const [isOpen, setIsOpen] = useState(false);

  function handleOpen() {
    setIsOpen(true);
    onOpen();
  }

  if (!isOpen) {
    return (
      <Button
        variant="link"
        size="small"
        disabled={isDisabled}
        onClick={handleOpen}
      >
        Start over
      </Button>
    );
  }

  return (
    <div
      role="group"
      aria-label="Start over"
      className="flex flex-col items-center gap-2"
    >
      {hasKept && (
        <Text variant="small" tone="muted" align="center">
          This replaces what you told Otto.
        </Text>
      )}
      <div className="flex flex-wrap items-center justify-center gap-2">
        {canCall && (
          <Button
            variant="secondary"
            size="small"
            disabled={isDisabled}
            onClick={onCallAgain}
          >
            Call again
          </Button>
        )}
        <Button
          variant="secondary"
          size="small"
          disabled={isDisabled}
          onClick={onTypeInstead}
        >
          Type instead
        </Button>
        <Button
          variant="ghost"
          size="small"
          disabled={isDisabled}
          onClick={() => setIsOpen(false)}
        >
          Never mind
        </Button>
      </div>
    </div>
  );
}
