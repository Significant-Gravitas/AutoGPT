import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { NeedsAttentionRow } from "./components/NeedsAttentionRow";
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
