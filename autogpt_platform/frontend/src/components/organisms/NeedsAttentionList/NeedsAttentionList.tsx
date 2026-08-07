import Link from "next/link";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { getReviewLink, getReviewTitle } from "./helpers";
import { useNeedsAttentionList } from "./useNeedsAttentionList";

export function NeedsAttentionList(): JSX.Element | null {
  const {
    reviews,
    count,
    isLoading,
    isError,
    refetch,
    isProcessing,
    approve,
    skip,
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

  if (count === 0) return null;

  return (
    <section className="flex flex-col gap-3">
      <Text variant="h5">Needs your attention ({count})</Text>
      <div className="flex flex-col gap-2">
        {reviews.map((review) => (
          <NeedsAttentionRow
            key={review.node_exec_id}
            review={review}
            isProcessing={isProcessing}
            onApprove={approve}
            onSkip={skip}
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
  onSkip: (review: PendingHumanReviewModel) => void;
}

function NeedsAttentionRow({
  review,
  isProcessing,
  onApprove,
  onSkip,
}: RowProps) {
  const subtitle = [review.expert_name, review.agent_name]
    .filter(Boolean)
    .join(" · ");

  return (
    <div className="flex items-center gap-3 rounded-xl border border-zinc-200 bg-white p-3">
      <ReviewAvatar review={review} />
      <Link href={getReviewLink(review)} className="min-w-0 flex-1">
        <Text variant="body-medium" className="truncate">
          {getReviewTitle(review)}
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
          onClick={() => onApprove(review)}
        >
          Approve
        </Button>
        <Button
          variant="ghost"
          size="small"
          disabled={isProcessing}
          onClick={() => onSkip(review)}
        >
          Skip
        </Button>
      </div>
    </div>
  );
}

function ReviewAvatar({ review }: { review: PendingHumanReviewModel }) {
  if (!review.expert_name) {
    return (
      <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-zinc-100">
        <Icon icon={Robot01Icon} size={20} className="text-zinc-500" />
      </div>
    );
  }

  return (
    <Avatar className="h-10 w-10 shrink-0">
      {review.expert_avatar_url ? (
        <AvatarImage src={review.expert_avatar_url} alt={review.expert_name} />
      ) : null}
      <AvatarFallback>{review.expert_name}</AvatarFallback>
    </Avatar>
  );
}
