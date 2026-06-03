"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { DeviceMobileIcon } from "@phosphor-icons/react";
import { useState } from "react";

export function MobileWarning() {
  const breakpoint = useBreakpoint();
  const [isDismissed, setIsDismissed] = useState(false);

  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const isOpen = isMobile && !isDismissed;

  return (
    <Dialog
      title="Builder works best on desktop"
      controlled={{
        isOpen,
        set: (next) => {
          if (!next) setIsDismissed(true);
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col items-center gap-4 px-1 py-2 text-center">
          <DeviceMobileIcon className="h-10 w-10 text-amber-600" />
          <Text variant="body" className="text-zinc-700">
            The agent builder relies on canvas interactions that don&apos;t work
            well on a phone. For the best experience, switch to a desktop
            browser.
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
      </Dialog.Content>
    </Dialog>
  );
}
