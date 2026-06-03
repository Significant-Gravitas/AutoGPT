"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Key, storage } from "@/services/storage/local-storage";
import { DeviceMobileIcon } from "@phosphor-icons/react";
import { useEffect, useState } from "react";

export function MobileWarning() {
  const breakpoint = useBreakpoint();
  const [isDismissed, setIsDismissed] = useState(false);
  const [isSuppressed, setIsSuppressed] = useState(true);

  useEffect(() => {
    setIsSuppressed(storage.get(Key.BUILDER_MOBILE_WARNING_SUPPRESSED) === "1");
  }, []);

  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const isOpen = isMobile && !isDismissed && !isSuppressed;

  function handleDontShowAgain() {
    storage.set(Key.BUILDER_MOBILE_WARNING_SUPPRESSED, "1");
    setIsSuppressed(true);
  }

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
            well on a small screen. For the best experience, switch to a desktop
            browser.
          </Text>
          <div className="mt-2 flex w-full flex-col gap-2 sm:flex-row sm:justify-center">
            <Button
              variant="secondary"
              size="small"
              onClick={() => setIsDismissed(true)}
            >
              Continue anyway
            </Button>
            <Button variant="ghost" size="small" onClick={handleDontShowAgain}>
              Don&apos;t show again
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
