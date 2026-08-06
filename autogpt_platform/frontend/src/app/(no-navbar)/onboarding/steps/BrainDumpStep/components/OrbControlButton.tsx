"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { SwapFade } from "@/components/atoms/SwapFade/SwapFade";
import {
  ArrowReloadHorizontalIcon,
  Mic01Icon,
} from "@hugeicons/core-free-icons";

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
          <Icon icon={ArrowReloadHorizontalIcon} size={22} strokeWidth={1.5} />
        ) : (
          <Icon icon={Mic01Icon} size={22} strokeWidth={1.5} />
        )}
      </SwapFade>
    </Button>
  );
}
