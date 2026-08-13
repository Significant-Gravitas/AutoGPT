"use client";

import { Text } from "@/components/atoms/Text/Text";
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

  return (
    <AnimatePresence>
      {selected ? (
        <motion.div
          role="dialog"
          aria-modal="true"
          aria-label={selected.profession.label}
          onClick={onClose}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: "easeOut" }}
          className="fixed inset-0 z-50 flex cursor-zoom-out flex-col items-center justify-center gap-6 bg-white/85 backdrop-blur-md"
        >
          <motion.div
            layoutId={prefersReducedMotion ? undefined : selected.layoutId}
            transition={EXPAND_TRANSITION}
            className="h-[22rem] w-[22rem] max-w-[80vw]"
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
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            transition={{ ...EXPAND_TRANSITION, delay: 0.05 }}
          >
            <Text variant="h3">{selected.profession.label}</Text>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
