"use client";

import { motion } from "framer-motion";
import { WarningCircleIcon } from "@phosphor-icons/react";

import { Text } from "@/components/atoms/Text/Text";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { ReviewStepper } from "./ReviewStepper";
import { ShareLinkButton } from "./ShareLinkButton";
import { ReviewHero } from "./ReviewHero";
import { SubmissionSummaryCard } from "./SubmissionSummaryCard";
import { ReviewStepFooter } from "./ReviewStepFooter";
import { useAgentReviewStep } from "./useAgentReviewStep";

interface Props {
  agentName: string;
  subheader: string;
  description: string;
  onClose: () => void;
  onDone: () => void;
  onViewProgress: () => void;
  onEdit?: () => void;
  thumbnailSrc?: string;
  status?: SubmissionStatus;
  reviewComments?: string | null;
  version?: number;
  category?: string | null;
  submittedAt?: string | Date | null;
  reviewedAt?: string | Date | null;
  runCount?: number;
  marketplaceUrl?: string;
}

export function AgentReviewStep({
  agentName,
  subheader,
  description: _description,
  thumbnailSrc,
  onDone,
  onViewProgress,
  onEdit,
  status,
  reviewComments,
  version,
  category,
  submittedAt,
  reviewedAt,
  runCount,
  marketplaceUrl,
}: Props) {
  const {
    isDashboardPage,
    hero,
    shouldReduceMotion,
    isApproved,
    isRejected,
    isDraft,
    isPending,
    showCelebration,
    showConfetti,
    metaItems,
  } = useAgentReviewStep({
    status,
    version,
    category,
    submittedAt,
    reviewedAt,
    runCount,
  });

  return (
    <div
      aria-labelledby="modal-title"
      className="relative flex flex-col items-center pb-4 pt-10"
    >
      <ReviewHero
        hero={hero}
        showCelebration={showCelebration}
        showConfetti={showConfetti}
        shouldReduceMotion={!!shouldReduceMotion}
      />

      <SubmissionSummaryCard
        agentName={agentName}
        subheader={subheader}
        thumbnailSrc={thumbnailSrc}
        isPending={isPending}
        shouldReduceMotion={!!shouldReduceMotion}
      />

      {metaItems.length > 0 ? (
        <motion.div
          initial={shouldReduceMotion ? { opacity: 0 } : { opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.24, ease: "easeOut", delay: 0.21 }}
          className="mt-4 grid w-full max-w-md grid-cols-2 gap-x-4 gap-y-3 rounded-[14px] border border-zinc-200 bg-zinc-50/60 p-4"
          data-testid="submission-meta"
        >
          {metaItems.map((item) => (
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
      ) : null}

      {reviewComments && status === SubmissionStatus.REJECTED ? (
        <div className="mt-4 w-full max-w-md rounded-[14px] border border-rose-200 bg-rose-50 p-3">
          <div className="mb-1 flex items-center gap-2 text-rose-700">
            <WarningCircleIcon size={16} weight="duotone" />
            <Text variant="small-medium" as="span" className="!text-current">
              Review feedback
            </Text>
          </div>
          <Text variant="small" className="text-rose-700">
            {reviewComments}
          </Text>
        </div>
      ) : null}

      {isPending ? (
        <ReviewStepper shouldReduceMotion={!!shouldReduceMotion} />
      ) : null}

      {isApproved && marketplaceUrl ? (
        <div className="mt-4 flex w-full max-w-md justify-center">
          <ShareLinkButton url={marketplaceUrl} />
        </div>
      ) : null}

      <ReviewStepFooter
        onDone={onDone}
        onViewProgress={onViewProgress}
        onEdit={onEdit}
        isApproved={isApproved}
        isRejected={isRejected}
        isDraft={isDraft}
        isPending={isPending}
        isDashboardPage={isDashboardPage}
        marketplaceUrl={marketplaceUrl}
      />
    </div>
  );
}
