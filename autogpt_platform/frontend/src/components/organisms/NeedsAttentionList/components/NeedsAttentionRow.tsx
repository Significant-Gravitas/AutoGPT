import Link from "next/link";
import { useState } from "react";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getReviewLink } from "@/lib/review-links";
import { getReviewTitle } from "../helpers";

interface Props {
  review: PendingHumanReviewModel;
  isProcessing: boolean;
  onApprove: (review: PendingHumanReviewModel) => void;
  onDecline: (review: PendingHumanReviewModel) => void;
}

export function NeedsAttentionRow({
  review,
  isProcessing,
  onApprove,
  onDecline,
}: Props) {
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
    <div
      data-slot="needs-attention-row"
      className="flex items-center gap-3 rounded-xl border border-border-base bg-surface p-3"
    >
      <ExpertAvatar
        name={review.expert_name ?? null}
        avatarUrl={review.expert_avatar_url ?? null}
      />
      <Link
        href={getReviewLink(review)}
        data-slot="needs-attention-row-link"
        className="min-w-0 flex-1"
      >
        <Text variant="body-medium" className="truncate">
          {title}
        </Text>
        {subtitle ? (
          <Text variant="small" tone="muted" className="truncate">
            {subtitle}
          </Text>
        ) : null}
      </Link>
      <div
        data-slot="needs-attention-row-actions"
        className="flex shrink-0 items-center gap-2"
      >
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
      <span
        aria-live="polite"
        data-slot="needs-attention-row-announcement"
        className="sr-only"
      >
        {isConfirmingDecline ? `Tap again to decline: ${title}` : ""}
      </span>
    </div>
  );
}
