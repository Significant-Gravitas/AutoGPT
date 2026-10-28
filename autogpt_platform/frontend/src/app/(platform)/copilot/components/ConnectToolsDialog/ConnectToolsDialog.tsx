"use client";

import { focusComposer } from "@/app/(platform)/components/FirstRunTour/focusStartingPoint";
import { getFocusableElements } from "@/app/(platform)/components/TabIntroCard/helpers";
import { isKeyIgnoringComposition } from "@/lib/keyboard";
import { useEffect, useRef } from "react";
import { createPortal } from "react-dom";
import { ConnectToolsPanel } from "./ConnectToolsPanel";

interface Props {
  isOpen: boolean;
  onClose: () => void;
}

// "Connect your tools" on its own, offered from the last step of the
// first-run walkthrough. Escape belongs to the panel: it steps back out of a
// provider before it closes anything, so a half-typed key is never lost.
export function ConnectToolsDialog({ isOpen, onClose }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const pressStartedOnBackdrop = useRef(false);

  useEffect(() => {
    if (!isOpen) return;
    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();

    // Containment, not a shortcut: Tab stays inside while an IME composes.
    function handleKeyDown(event: KeyboardEvent) {
      if (!isKeyIgnoringComposition(event, "Tab")) return;
      const dialog = dialogRef.current;
      if (!dialog) return;
      const focusable = getFocusableElements(dialog);
      if (focusable.length === 0) return;
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (!dialog.contains(document.activeElement)) {
        event.preventDefault();
        first.focus();
      } else if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      // Opened from the walkthrough's last step, whose button is gone by
      // now: the composer is where a keyboard user carries on.
      if (
        previouslyFocused &&
        previouslyFocused !== document.body &&
        document.contains(previouslyFocused)
      )
        previouslyFocused.focus();
      else focusComposer();
    };
  }, [isOpen]);

  if (!isOpen) return null;

  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-zinc-950/30 px-4"
      // Only a press that starts on the backdrop closes it: a text selection
      // dragged out of the dialog ends with a click here too.
      onPointerDown={(event) => {
        pressStartedOnBackdrop.current = event.target === event.currentTarget;
      }}
      onClick={(event) => {
        if (
          event.target === event.currentTarget &&
          pressStartedOnBackdrop.current
        )
          onClose();
        pressStartedOnBackdrop.current = false;
      }}
    >
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label="Connect your tools"
        tabIndex={-1}
        className="w-full max-w-[30rem] overflow-hidden rounded-xl border border-border-base bg-surface shadow-lg outline-none"
      >
        <ConnectToolsPanel
          onBack={onClose}
          onNext={onClose}
          backLabel="Not now"
          nextLabel="Done"
        />
      </div>
    </div>,
    document.body,
  );
}
