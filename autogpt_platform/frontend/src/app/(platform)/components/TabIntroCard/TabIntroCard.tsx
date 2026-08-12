"use client";

import { GlassPixelBackdrop } from "@/components/atoms/GlassPixelBackdrop/GlassPixelBackdrop";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { IconSvgElement } from "@hugeicons/react";
import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useRef } from "react";
import { getFocusableElements } from "./helpers";

interface Action {
  label: string;
  onClick: () => void;
}

interface Props {
  isOpen: boolean;
  icon: IconSvgElement;
  title: string;
  /** One sentence under the title. */
  body: string;
  /** The card's primary way forward, sat next to "Got it". */
  cta: Action;
  /** Optional quiet alternative under the copy (Build's DIY route). */
  altAction?: Action;
  /** "Got it", Escape, and a click on the backdrop all land here. */
  onDismiss: () => void;
}

// One-card, first-visit intro for a tab. Deliberately the same shell as the
// copilot home's capability cards (OnboardingWelcomeDialog): tinted stage
// carrying the icon, copy below, one way forward and one way out.
export function TabIntroCard({
  isOpen,
  icon,
  title,
  body,
  cta,
  altAction,
  onDismiss,
}: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);

  // The listener below is installed once per open, so it would otherwise
  // capture the `onDismiss` from that render. A ref keeps it on the current
  // one without tearing the listener down on every parent render.
  const dismissRef = useRef(onDismiss);
  useEffect(() => {
    dismissRef.current = onDismiss;
  });

  // Focus starts inside the card rather than on whatever was behind the
  // overlay, stays there while the dialog is open (`aria-modal` promises as
  // much), and goes back where it came from on close. Escape is a dismissal
  // like every other way out.
  useEffect(() => {
    if (!isOpen) return;

    const previouslyFocused = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        dismissRef.current();
        return;
      }
      if (event.key !== "Tab") return;

      const dialog = dialogRef.current;
      if (!dialog) return;

      const focusable = getFocusableElements(dialog);
      if (focusable.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;

      if (!dialog.contains(active)) {
        event.preventDefault();
        (event.shiftKey ? last : first).focus();
      } else if (event.shiftKey && active === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      // Skip a node the close itself removed — refocusing it does nothing but
      // hand focus back to <body>.
      if (previouslyFocused && document.contains(previouslyFocused)) {
        previouslyFocused.focus();
      }
    };
  }, [isOpen]);

  return (
    <AnimatePresence>
      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.4, ease: [0.4, 0, 0.2, 1] }}
          // Only the backdrop itself dismisses — a click that started on the
          // card and drifted out (text selection) must not close it.
          onClick={(event) => {
            if (event.target === event.currentTarget) onDismiss();
          }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-white/30 px-4 backdrop-blur-sm"
          data-testid="tab-intro-overlay"
          role="dialog"
          aria-modal="true"
          aria-label={title}
        >
          <motion.div
            initial={{ opacity: 0, y: 16, scale: 0.97 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ duration: 0.45, ease: [0, 0, 0.2, 1] }}
            className="w-full max-w-[26rem] overflow-hidden rounded-3xl bg-white shadow-[0_24px_80px_-24px_rgba(0,0,0,0.3)] outline-none"
            ref={dialogRef}
            tabIndex={-1}
          >
            {/* Tinted stage: the card's icon floats here. */}
            <div className="relative h-44 bg-gradient-to-br from-[#e6dbff] via-[#ddccff] to-[#d0b9ff]">
              <GlassPixelBackdrop />
              <div className="flex h-full items-center justify-center">
                <div className="flex h-20 w-20 items-center justify-center rounded-3xl bg-white shadow-lg">
                  <Icon icon={icon} size={40} className="text-violet-600" />
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-3 px-7 pb-7 pt-6 text-left">
              <Text variant="lead-semibold" as="h3" className="text-zinc-900">
                {title}
              </Text>
              <Text variant="large" className="text-zinc-600">
                {body}
              </Text>
              {altAction && (
                <Button
                  variant="ghost"
                  size="small"
                  onClick={altAction.onClick}
                  className="w-fit self-start px-0 text-violet-600 underline-offset-4 hover:bg-transparent hover:underline"
                >
                  {altAction.label}
                </Button>
              )}

              <div className="mt-3 flex items-center justify-end gap-3">
                <Button variant="secondary" size="small" onClick={onDismiss}>
                  Got it
                </Button>
                <Button variant="primary" size="small" onClick={cta.onClick}>
                  {cta.label}
                </Button>
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
