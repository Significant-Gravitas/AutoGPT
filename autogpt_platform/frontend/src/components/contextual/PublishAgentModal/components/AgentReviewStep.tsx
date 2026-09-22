"use client";

import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { ReviewStepper } from "./ReviewStepper";
import { ShareLinkButton } from "./ShareLinkButton";
import { ReviewHero } from "./ReviewHero";
import { SubmissionSummaryCard } from "./SubmissionSummaryCard";
import { SubmissionMetaGrid } from "./SubmissionMetaGrid";
import { RejectionFeedback } from "./RejectionFeedback";
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

      <SubmissionMetaGrid
        items={metaItems}
        shouldReduceMotion={!!shouldReduceMotion}
      />

      {reviewComments && status === SubmissionStatus.REJECTED ? (
        <RejectionFeedback comments={reviewComments} />
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
