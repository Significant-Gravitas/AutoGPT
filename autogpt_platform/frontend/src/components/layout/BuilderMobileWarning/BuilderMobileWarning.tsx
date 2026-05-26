"use client";

import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { DeviceMobileIcon, XIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { Text } from "../../atoms/Text/Text";

export function BuilderMobileWarning() {
  const breakpoint = useBreakpoint();
  const isMobile = breakpoint === "base" || breakpoint === "sm";
  const [isDismissed, setIsDismissed] = useState(false);

  if (!isMobile || isDismissed) {
    return null;
  }

  return (
    <div className="absolute inset-x-0 top-0 z-50 mx-4 mt-4 rounded-lg border border-amber-200 bg-amber-50 p-4 shadow-sm">
      <div className="flex items-start gap-3">
        <DeviceMobileIcon className="mt-0.5 h-5 w-5 flex-shrink-0 text-amber-600" />
        <div className="flex flex-1 flex-col gap-1">
          <Text variant="body-medium" className="text-amber-900">
            Builder requires a desktop browser
          </Text>
          <Text variant="small" className="text-amber-800">
            The graph builder uses canvas interactions that work best with a
            mouse. For the best experience, open this page on a desktop or
            laptop.
          </Text>
        </div>
        <button
          onClick={() => setIsDismissed(true)}
          aria-label="Dismiss warning"
          className="flex-shrink-0 rounded p-0.5 text-amber-700 hover:bg-amber-100"
        >
          <XIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
}
