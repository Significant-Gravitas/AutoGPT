"use client";

import * as Dialog from "@radix-ui/react-dialog";
import { PropsWithChildren, useRef } from "react";

interface Props extends PropsWithChildren {
  title: string;
  onClose: () => void;
}

export function FullscreenDialog({ title, onClose, children }: Props) {
  const previousFocus = useRef<HTMLElement | null>(null);
  return (
    <Dialog.Root
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-background" />
        <Dialog.Content
          className="fixed inset-0 z-50 flex flex-col bg-background"
          aria-describedby={undefined}
          onOpenAutoFocus={() => {
            previousFocus.current =
              document.activeElement instanceof HTMLElement
                ? document.activeElement
                : null;
          }}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            if (previousFocus.current?.isConnected)
              previousFocus.current.focus();
          }}
        >
          <Dialog.Title className="sr-only">{title}</Dialog.Title>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
