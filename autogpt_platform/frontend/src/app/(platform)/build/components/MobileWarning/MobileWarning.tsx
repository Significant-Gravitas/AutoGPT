"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { DeviceMobile } from "@phosphor-icons/react";
import { useState } from "react";

export function MobileWarning() {
  const breakpoint = useBreakpoint();
  const [isDismissed, setIsDismissed] = useState(false);

  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  if (!isMobile || isDismissed) {
    return null;
  }

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-labelledby="builder-mobile-warning-title"
      className="absolute inset-0 z-50 flex items-center justify-center bg-zinc-50/95 p-6 backdrop-blur-sm"
    >
      <div className="flex w-full max-w-md flex-col items-center gap-4 rounded-lg border border-amber-200 bg-amber-50 p-6 shadow-lg">
        <DeviceMobile className="h-10 w-10 text-amber-600" />
        <Text
          variant="h3"
          id="builder-mobile-warning-title"
          className="text-center text-amber-900"
        >
          Builder works best on desktop
        </Text>
        <Text variant="body" className="text-center text-amber-800">
          The agent builder relies on canvas interactions that don&apos;t work
          well on a phone. For the best experience, switch to a desktop browser.
        </Text>
        <Button
          variant="secondary"
          size="small"
          onClick={() => setIsDismissed(true)}
          className="mt-2"
        >
          Continue anyway
        </Button>
      </div>
    </div>
  );
}
