"use client";

import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { DeviceMobile } from "@phosphor-icons/react";
import { Text } from "../atoms/Text/Text";

export function MobileWarningBanner() {
  const breakpoint = useBreakpoint();
  const isMobile = breakpoint === "base" || breakpoint === "sm";

  if (!isMobile) {
    return null;
  }

  return (
    <div className="mx-auto mt-6 w-full max-w-[32rem] rounded-lg border border-amber-200 bg-amber-50 p-4">
      <div className="flex items-start gap-3">
        <DeviceMobile className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-600" />
        <div className="flex flex-col gap-1">
          <Text variant="body-medium" className="text-amber-900">
            Heads up: AutoGPT works best on desktop
          </Text>
          <Text variant="small" className="text-amber-800">
            Some features may be limited on mobile. For the best experience,
            consider switching to a desktop.
          </Text>
        </div>
      </div>
    </div>
  );
}
