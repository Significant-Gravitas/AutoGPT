"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { motion } from "framer-motion";
import { Text } from "@/components/atoms/Text/Text";
import { Clock01Icon, ImageNotFound01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      className="mt-6 flex w-full max-w-md items-center gap-3 rounded-xl border border-border-base bg-surface p-3 shadow-subtle"
    >
      <div className="relative aspect-video h-12 shrink-0 overflow-hidden rounded-lg bg-surface-hover">
        {thumbnailSrc ? (
          <Image
            src={thumbnailSrc}
            unoptimized={isLocalStoreMediaUrl(thumbnailSrc)}
            alt=""
            fill
            sizes="86px"
            className="object-cover"
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-foreground-subtle">
            <Icon icon={ImageNotFound01Icon} size={20} />
          </div>
        )}
      </div>
      <div className="flex min-w-0 flex-1 flex-col">
        <Text variant="body-medium" as="span" className="truncate">
          {agentName}
        </Text>
        {subheader ? (
          <Text variant="small" tone="muted" className="truncate">
            {subheader}
          </Text>
        ) : null}
      </div>
      {isPending ? (
        <span className="inline-flex items-center gap-1.5 rounded-full bg-warning-surface px-2.5 py-1 text-caption-sm font-medium leading-inherit text-warning-foreground">
          <Icon icon={Clock01Icon} size={12} />
          In review
        </span>
      ) : null}
    </motion.div>
  );
}
