"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Mic01Icon, PencilEdit01Icon } from "@hugeicons/core-free-icons";

interface Props {
  onTalk: () => void;
  onWrite: () => void;
}

// The two ways to tell Otto about your work, side by side, so neither
// is hidden behind a link. Talking leads; writing is the equal alternative.
export function RestActions({ onTalk, onWrite }: Props) {
  return (
    <div className="mt-4 flex items-center gap-3">
      <Button
        type="button"
        size="small"
        onClick={onTalk}
        leadingIcon={Mic01Icon}
        className="h-10 w-36 rounded-xl"
      >
        Talk
      </Button>
      <Button
        type="button"
        variant="secondary"
        size="small"
        onClick={onWrite}
        leadingIcon={PencilEdit01Icon}
        className="h-10 w-36 rounded-xl"
      >
        Write
      </Button>
    </div>
  );
}
