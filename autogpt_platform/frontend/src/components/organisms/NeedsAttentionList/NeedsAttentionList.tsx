import Link from "next/link";
import { useState } from "react";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getReviewLink } from "@/lib/review-links";
import { getReviewTitle } from "./helpers";
import { useNeedsAttentionList } from "./useNeedsAttentionList";

export function NeedsAttentionList() {
  const {
    reviews,
    countLabel,
    isLoading,
    isError,
    refetch,
    pendingNodeExecIds,
    approve,
    decline,
  } = useNeedsAttentionList();

  if (isLoading) {
    return (
      <div className="flex flex-col gap-2">
        <Skeleton className="h-5 w-40" />
        <Skeleton className="h-16 w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        context="pending reviews"
        httpError={{ message: "Failed to load pending reviews" }}
        onRetry={() => refetch()}
      />
    );
  }

  if (reviews.length === 0) return null;

  return (
    <section className="flex flex-col gap-3">
      <Text variant="h5">Needs your attention ({countLabel})</Text>
      <div className="flex flex-col gap-2">
        {reviews.map((review) => (
          <NeedsAttentionRow
            key={review.node_exec_id}
            review={review}
            // Only the acted row locks up, so morning triage stays
            // parallel instead of serializing on each round-trip.
            isProcessing={pendingNodeExecIds.has(review.node_exec_id)}
            onApprove={approve}
            onDecline={decline}
          />
        ))}
      </div>
    </section>
  );
}

interface RowProps {
  review: PendingHumanReviewModel;
  isProcessing: boolean;
  onApprove: (review: PendingHumanReviewModel) => void;
  onDecline: (review: PendingHumanReviewModel) => void;
}

function NeedsAttentionRow({
  review,
  isProcessing,
  onApprove,
  onDecline,
}: RowProps) {
  // Decline rejects the agent's action outright, with no undo, in a flow
  // built for fast tapping — so it takes a second, deliberate tap.
  const [isConfirmingDecline, setIsConfirmingDecline] = useState(false);
  const title = getReviewTitle(review);
  const subtitle = [review.expert_name, review.agent_name]
    .filter(Boolean)
    .join(" · ");

  function handleDecline() {
    if (!isConfirmingDecline) {
      setIsConfirmingDecline(true);
      return;
    }
    setIsConfirmingDecline(false);
    onDecline(review);
  }

  return (
    <div className="flex items-center gap-3 rounded-xl border border-zinc-200 bg-white p-3">
      <ExpertAvatar
        name={review.expert_name ?? null}
        avatarUrl={review.expert_avatar_url ?? null}
      />
      <Link href={getReviewLink(review)} className="min-w-0 flex-1">
        <Text variant="body-medium" className="truncate">
          {title}
        </Text>
        {subtitle ? (
          <Text variant="small" className="truncate text-zinc-500">
            {subtitle}
          </Text>
        ) : null}
      </Link>
      <div className="flex shrink-0 items-center gap-2">
        <Button
          variant="primary"
          size="small"
          disabled={isProcessing}
          aria-label={`Approve: ${title}`}
          onClick={() => onApprove(review)}
        >
          Approve
        </Button>
        <Button
          // The armed state carries the destructive colour as well as the
          // label: in a fast-tapping flow a word swap alone is easy to miss,
          // and the next tap has no undo.
          variant={isConfirmingDecline ? "destructive" : "ghost"}
          size="small"
          disabled={isProcessing}
          aria-label={
            isConfirmingDecline
              ? `Confirm decline: ${title}`
              : `Decline: ${title}`
          }
          onClick={handleDecline}
          onBlur={() => setIsConfirmingDecline(false)}
        >
          {isConfirmingDecline ? "Confirm" : "Decline"}
        </Button>
      </div>
      <span aria-live="polite" className="sr-only">
        {isConfirmingDecline ? `Tap again to decline: ${title}` : ""}
      </span>
    </div>
  );
}
