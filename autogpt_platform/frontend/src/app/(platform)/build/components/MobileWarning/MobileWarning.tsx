"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { DeviceMobileIcon } from "@phosphor-icons/react";
import { useMobileWarning } from "./useMobileWarning";

export function MobileWarning() {
  const { isOpen, dismiss, suppress } = useMobileWarning();

  return (
    <Dialog
      title="Builder works best on desktop"
      controlled={{
        isOpen,
        set: (next) => {
          if (!next) dismiss();
        },
      }}
    >
      <Dialog.Content>
        <div className="flex flex-col items-center gap-4 px-1 py-2 text-center">
          <DeviceMobileIcon className="h-10 w-10 text-amber-600" />
          <Text variant="body" className="text-zinc-700">
            The agent builder relies on canvas interactions that don&apos;t work
            well on this screen size. For the best experience, switch to a
            desktop browser.
          </Text>
          <div className="mt-2 flex w-full flex-col gap-2 sm:flex-row sm:justify-center">
            <Button variant="secondary" size="small" onClick={dismiss}>
              Continue anyway
            </Button>
            <Button variant="ghost" size="small" onClick={suppress}>
              Don&apos;t show again
            </Button>
          </div>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
