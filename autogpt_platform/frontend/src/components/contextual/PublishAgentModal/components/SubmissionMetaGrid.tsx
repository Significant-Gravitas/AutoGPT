"use client";

import { motion } from "framer-motion";

import { Text } from "@/components/atoms/Text/Text";
import type { SubmissionMetaItem } from "../helpers";

interface Props {
  items: SubmissionMetaItem[];
  shouldReduceMotion: boolean;
}

export function SubmissionMetaGrid({ items, shouldReduceMotion }: Props) {
  if (items.length === 0) return null;

  return (
    <motion.div
      initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24, ease: "easeOut", delay: 0.21 }}
      className="mt-4 grid w-full max-w-md grid-cols-2 gap-x-4 gap-y-3 rounded-[14px] border border-zinc-200 bg-zinc-50/60 p-4"
      data-testid="submission-meta"
    >
      {items.map((item) => (
        <div key={item.label} className="flex min-w-0 flex-col">
          <Text variant="small" as="span" className="text-zinc-500">
            {item.label}
          </Text>
          <Text
            variant="small-medium"
            as="span"
            title={item.title}
            className="truncate text-textBlack"
          >
            {item.value}
          </Text>
        </div>
      ))}
    </motion.div>
  );
}
