"use client";

import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Text } from "../atoms/Text/Text";
import { SmartPhone01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function MobileWarningBanner() {
  const breakpoint = useBreakpoint();
  const isMobile = breakpoint === "base" || breakpoint === "sm";

  if (!isMobile) {
    return null;
  }

  return (
    <div className="mx-auto mt-6 w-full max-w-[32rem] rounded-lg border border-warning-border bg-warning-surface p-4">
      <div className="flex items-start gap-3">
        <Icon
          icon={SmartPhone01Icon}
          className="mt-0.5 h-5 w-5 flex-shrink-0 text-warning-foreground"
        />
        <div className="flex flex-col gap-1">
          <Text variant="body-medium" tone="warning">
            Heads up: AutoGPT works best on desktop
          </Text>
          <Text variant="small" tone="warning">
            Some features may be limited on mobile. For the best experience,
            consider switching to a desktop.
          </Text>
        </div>
      </div>
    </div>
  );
}
