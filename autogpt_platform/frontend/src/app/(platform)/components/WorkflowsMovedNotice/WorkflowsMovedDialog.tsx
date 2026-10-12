"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { isComposingEvent } from "@/lib/keyboard";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import * as Dialog from "@radix-ui/react-dialog";
import { useRef } from "react";
import { WorkflowsMovedContent } from "./WorkflowsMovedContent";
import { hasCompetingDialog } from "./helpers";

interface Props {
  isOpen: boolean;
  onDismiss: () => void;
}

export function WorkflowsMovedDialog({ isOpen, onDismiss }: Props) {
  const titleRef = useRef<HTMLHeadingElement>(null);
  const previousFocus = useRef<HTMLElement | null>(null);
  return (
    <Dialog.Root open={isOpen} onOpenChange={(open) => !open && onDismiss()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-[100] bg-zinc-950/25 backdrop-blur-sm motion-safe:animate-fade-in" />
        <Dialog.Content
          data-workflows-moved-notice=""
          className="fixed left-1/2 top-1/2 z-[100] flex max-h-[calc(100dvh-2rem)] w-[calc(100%-2rem)] max-w-[44rem] -translate-x-1/2 -translate-y-1/2 flex-col overflow-hidden rounded-3xl border border-white/80 bg-white shadow-2xl outline-none motion-safe:animate-fade-in"
          onEscapeKeyDown={(event) => {
            if (isComposingEvent(event)) event.preventDefault();
          }}
          onOpenAutoFocus={(event) => {
            event.preventDefault();
            previousFocus.current =
              document.activeElement instanceof HTMLElement
                ? document.activeElement
                : null;
            titleRef.current?.focus();
          }}
          onCloseAutoFocus={(event) => {
            event.preventDefault();
            if (hasCompetingDialog()) return;
            if (previousFocus.current?.isConnected)
              previousFocus.current.focus();
          }}
        >
          <Button
            variant="icon"
            size="icon-sm"
            aria-label="Close notice"
            onClick={onDismiss}
            withTooltip={false}
            className="absolute right-4 top-4 z-10 rounded-full border-white/80 bg-white/80"
          >
            <Icon icon={Cancel01Icon} size={16} aria-hidden />
          </Button>
          <WorkflowsMovedContent titleRef={titleRef} onDismiss={onDismiss} />
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
