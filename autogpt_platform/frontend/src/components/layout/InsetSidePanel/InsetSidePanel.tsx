"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { FullscreenDialog } from "@/components/molecules/FullscreenDialog/FullscreenDialog";
import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { SidebarRightIcon } from "@hugeicons/core-free-icons";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { CSSProperties, ReactNode, useSyncExternalStore } from "react";
import { createPortal } from "react-dom";
import { InsetPanelBody } from "./components/InsetPanelBody";

const SLOT_ID = "inset-side-panel-slot";
const EASE_OUT_QUART: [number, number, number, number] = [0.25, 1, 0.5, 1];
const MOBILE_BREAKPOINTS = new Set(["base", "sm", "md"]);

export function InsetSideSlot() {
  return <div id={SLOT_ID} className="contents" />;
}

interface PortalProps {
  children: ReactNode;
}

function subscribe() {
  return () => {};
}

/** The page's side slot, or null on a page whose frame has none. */
export function useInsetSideSlot() {
  return useSyncExternalStore(
    subscribe,
    () => document.getElementById(SLOT_ID),
    () => null,
  );
}

export function InsetSidePortal({ children }: PortalProps) {
  const slot = useInsetSideSlot();

  return slot ? createPortal(children, slot) : null;
}

interface Props {
  isOpen: boolean;
  title: string;
  ariaLabel: string;
  closeLabel: string;
  onClose: () => void;
  width?: number;
  children: ReactNode;
}

export function InsetSidePanel({
  isOpen,
  title,
  ariaLabel,
  closeLabel,
  onClose,
  width = 380,
  children,
}: Props) {
  const isMobile = MOBILE_BREAKPOINTS.has(useBreakpoint());
  const reduceMotion = useReducedMotion();

  if (isMobile) {
    if (!isOpen) return null;
    return (
      <FullscreenDialog title={ariaLabel} onClose={onClose}>
        {children}
      </FullscreenDialog>
    );
  }

  const slide = reduceMotion ? 0 : 16;
  const speed = reduceMotion ? 0 : 1;

  return (
    <InsetSidePortal>
      <AnimatePresence initial={false}>
        {isOpen && (
          <motion.aside
            aria-label={ariaLabel}
            initial={{ width: 0 }}
            animate={{ width }}
            exit={{ width: 0 }}
            transition={{ duration: 0.42 * speed, ease: EASE_OUT_QUART }}
            className="sticky top-0 h-svh shrink-0 self-start overflow-hidden"
          >
            <motion.div
              initial={{ opacity: 0, x: slide }}
              animate={{
                opacity: 1,
                x: 0,
                transition: {
                  opacity: { duration: 0.26 * speed, ease: EASE_OUT_QUART },
                  x: { duration: 0.42 * speed, ease: EASE_OUT_QUART },
                },
              }}
              exit={{
                opacity: 0,
                x: slide,
                transition: {
                  opacity: { duration: 0.2 * speed, ease: EASE_OUT_QUART },
                  x: { duration: 0.42 * speed, ease: EASE_OUT_QUART },
                },
              }}
              style={{ "--inset-panel-width": `${width}px` } as CSSProperties}
              className="flex h-full w-(--inset-panel-width) flex-col py-2.5 pr-2.5"
            >
              <header className="flex h-7 shrink-0 items-center justify-between pl-1.5">
                <Text variant="body-medium" as="h2" tone="ink">
                  {title}
                </Text>
                <Button
                  type="button"
                  variant="ghost-muted"
                  size="icon-sm"
                  leadingIcon={SidebarRightIcon}
                  aria-label={closeLabel}
                  className="size-7"
                  onClick={onClose}
                />
              </header>
              <InsetPanelBody>{children}</InsetPanelBody>
            </motion.div>
          </motion.aside>
        )}
      </AnimatePresence>
    </InsetSidePortal>
  );
}
