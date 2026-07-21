"use client";

import Image from "next/image";
import { motion } from "framer-motion";
import { ClockIcon, ImageBrokenIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  agentName: string;
  subheader: string;
  thumbnailSrc?: string;
  isPending: boolean;
  shouldReduceMotion: boolean;
}

export function SubmissionSummaryCard({
  agentName,
  subheader,
  thumbnailSrc,
  isPending,
  shouldReduceMotion,
}: Props) {
  return (
    <motion.div
      initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.24, ease: "easeOut", delay: 0.18 }}
      className="mt-6 flex w-full max-w-md items-center gap-3 rounded-[14px] border border-zinc-200 bg-white p-3 shadow-[0_1px_2px_rgba(15,15,20,0.04)]"
    >
      <div className="relative aspect-video h-12 shrink-0 overflow-hidden rounded-[8px] bg-zinc-100">
        {thumbnailSrc ? (
          <Image
            src={thumbnailSrc}
            alt=""
            fill
            sizes="86px"
            className="object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-zinc-400">
            <ImageBrokenIcon size={20} weight="duotone" />
          </div>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <Text
          variant="body-medium"
          as="span"
          className="truncate text-textBlack"
        >
          {agentName}
        </Text>
        {subheader ? (
          <Text variant="small" className="truncate text-zinc-500">
            {subheader}
          </Text>
        ) : null}
      </div>
      {isPending ? (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-amber-50 px-2.5 py-1 text-[11px] font-medium text-amber-800">
          <ClockIcon size={12} weight="duotone" />
          In review
        </span>
      ) : null}
    </motion.div>
  );
}
