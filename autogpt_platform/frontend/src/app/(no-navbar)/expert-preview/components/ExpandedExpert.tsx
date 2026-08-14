"use client";

import { Text } from "@/components/atoms/Text/Text";
import * as Dialog from "@radix-ui/react-dialog";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import Image from "next/image";
import { getProfessionImageSrc } from "../helpers";
import { SelectedExpert } from "../useExpertPreview";

interface Props {
  selected: SelectedExpert | null;
  onClose: () => void;
}

// iOS sheet curve — decelerates hard so the avatar settles instead of landing flat.
const EXPAND_TRANSITION = { duration: 0.3, ease: [0.32, 0.72, 0, 1] as const };

export function ExpandedExpert({ selected, onClose }: Props) {
  const prefersReducedMotion = useReducedMotion();
  const backdropTransition = prefersReducedMotion
    ? { duration: 0 }
    : { duration: 0.2, ease: "easeOut" as const };
  const contentTransition = prefersReducedMotion
    ? { duration: 0 }
    : EXPAND_TRANSITION;

  return (
    <Dialog.Root
      open={selected !== null}
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <AnimatePresence>
        {selected ? (
          <Dialog.Portal forceMount>
            <Dialog.Content
              asChild
              forceMount
              aria-describedby={undefined}
              onCloseAutoFocus={(event) => {
                event.preventDefault();
                selected.trigger.focus();
              }}
            >
              <motion.div
                initial={prefersReducedMotion ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={prefersReducedMotion ? { opacity: 1 } : { opacity: 0 }}
                transition={backdropTransition}
                className="fixed inset-0 z-50 flex cursor-zoom-out flex-col items-center justify-center gap-6 bg-white/85 backdrop-blur-md"
              >
                <Dialog.Close asChild>
                  <button
                    type="button"
                    data-testid="expert-preview-backdrop"
                    aria-label={`Close ${selected.profession.label} preview`}
                    className="absolute inset-0 cursor-zoom-out"
                  />
                </Dialog.Close>
                <motion.div
                  layoutId={
                    prefersReducedMotion ? undefined : selected.layoutId
                  }
                  transition={contentTransition}
                  className="pointer-events-none relative z-10 h-[22rem] w-[22rem] max-w-[80vw]"
                >
                  <Image
                    src={getProfessionImageSrc(selected.profession.slug)}
                    alt={selected.profession.label}
                    width={1024}
                    height={1024}
                    className="h-full w-full object-contain"
                    priority
                  />
                </motion.div>
                <Dialog.Title asChild>
                  <motion.div
                    initial={
                      prefersReducedMotion ? false : { opacity: 0, y: 8 }
                    }
                    animate={{ opacity: 1, y: 0 }}
                    exit={
                      prefersReducedMotion
                        ? { opacity: 1, y: 0 }
                        : { opacity: 0 }
                    }
                    transition={
                      prefersReducedMotion
                        ? { duration: 0 }
                        : { ...EXPAND_TRANSITION, delay: 0.05 }
                    }
                    className="pointer-events-none relative z-10"
                  >
                    <Text variant="h3">{selected.profession.label}</Text>
                  </motion.div>
                </Dialog.Title>
              </motion.div>
            </Dialog.Content>
          </Dialog.Portal>
        ) : null}
      </AnimatePresence>
    </Dialog.Root>
  );
}
