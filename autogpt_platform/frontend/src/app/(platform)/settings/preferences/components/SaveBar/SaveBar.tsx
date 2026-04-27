"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import { EASE_IOS } from "../../helpers";

interface Props {
  visible: boolean;
  saving: boolean;
  onDiscard: () => void;
  onSave: () => void;
}

export function SaveBar({ visible, saving, onDiscard, onSave }: Props) {
  const reduceMotion = useReducedMotion();

  return (
    <AnimatePresence initial={false}>
      {visible ? (
        <motion.div
          key="preferences-save-bar"
          initial={
            reduceMotion ? { opacity: 0 } : { opacity: 0, y: 28, scale: 0.98 }
          }
          animate={{ opacity: 1, y: 0, scale: 1 }}
          exit={
            reduceMotion ? { opacity: 0 } : { opacity: 0, y: 28, scale: 0.98 }
          }
          transition={{ duration: 0.28, ease: EASE_IOS }}
          className="pointer-events-none fixed inset-x-0 bottom-4 z-30 flex justify-center px-4"
        >
          <div className="pointer-events-auto flex w-full max-w-[640px] items-center justify-between gap-4 rounded-[14px] border border-zinc-200 bg-white/95 px-4 py-3 shadow-[0_12px_30px_-10px_rgba(15,15,20,0.18)] backdrop-blur-md">
            <div className="flex min-w-0 flex-col">
              <Text variant="large-medium" className="text-[#1F1F20]">
                Unsaved changes
              </Text>
              <Text variant="small" className="text-zinc-500">
                Review your time zone or notifications, then save.
              </Text>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <Button
                variant="ghost"
                size="small"
                onClick={onDiscard}
                disabled={saving}
              >
                Discard
              </Button>
              <Button
                variant="primary"
                size="small"
                onClick={onSave}
                loading={saving}
                disabled={saving}
              >
                Save changes
              </Button>
            </div>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
