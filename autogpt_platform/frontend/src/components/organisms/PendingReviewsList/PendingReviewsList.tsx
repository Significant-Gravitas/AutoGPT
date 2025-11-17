import { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { PendingReviewCard } from "@/components/organisms/PendingReviewCard/PendingReviewCard";
import { Text } from "@/components/atoms/Text/Text";
import { ClockIcon } from "@phosphor-icons/react";

interface PendingReviewsListProps {
  reviews: PendingHumanReviewModel[];
  onReviewComplete?: () => void;
  emptyMessage?: string;
}

export function PendingReviewsList({
  reviews,
  onReviewComplete,
  emptyMessage = "No pending reviews",
}: PendingReviewsListProps) {
  if (reviews.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-center">
        <ClockIcon size={48} className="mb-4 text-muted-foreground" />
        <Text variant="h4" className="text-muted-foreground">
          {emptyMessage}
        </Text>
        <Text variant="body" className="mt-2 max-w-md text-muted-foreground">
          When agents have human-in-the-loop blocks, they will appear here for
          your review and approval.
        </Text>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-2">
        <ClockIcon size={20} className="text-blue-600" />
        <Text variant="h3">Pending Reviews ({reviews.length})</Text>
      </div>

      <div className="space-y-4">
        {reviews.map((review) => (
          <PendingReviewCard
            key={review.node_exec_id}
            review={review}
            onReviewComplete={onReviewComplete}
          />
        ))}
      </div>
    </div>
  );
}
