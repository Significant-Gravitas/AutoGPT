"use client";

import { Button } from "@/components/atoms/Button/Button";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import {
  ArrowCounterClockwiseIcon,
  MicrophoneIcon,
} from "@phosphor-icons/react";

interface Props {
  screen: "rest" | "failed";
  onClick?: () => void;
}

export function OrbControlButton({ screen, onClick }: Props) {
  const ariaLabel = screen === "failed" ? "Try again" : "Start talking";

  return (
    <Button
      variant="icon"
      size="icon"
      onClick={onClick}
      aria-label={ariaLabel}
      className="mt-4 border border-black/5 bg-white shadow-sm hover:border-black/5 hover:bg-zinc-50"
    >
      <SwapFade swapKey={screen} className="flex items-center justify-center">
        {screen === "failed" ? (
          <ArrowCounterClockwiseIcon size={22} weight="light" />
        ) : (
          <MicrophoneIcon size={22} weight="light" />
        )}
      </SwapFade>
    </Button>
  );
}
