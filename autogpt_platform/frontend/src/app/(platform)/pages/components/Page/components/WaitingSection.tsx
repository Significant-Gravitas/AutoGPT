import type { PageWaiting } from "@/app/api/__generated__/models/pageWaiting";
import { Text } from "@/components/atoms/Text/Text";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import type { PageLive } from "../Page";

interface Props {
  section: PageWaiting;
  live?: PageLive;
}

export function WaitingSection({ section, live }: Props) {
  if (!section.reviews.length) {
    return (
      <Text variant="body" tone="secondary">
        Nothing is waiting on you.
      </Text>
    );
  }
  if (!live) {
    return (
      <Text variant="body">
        {section.reviews.length === 1
          ? "One approval is waiting on you."
          : `${section.reviews.length} approvals are waiting on you.`}
      </Text>
    );
  }
  return (
    <PendingReviewsList
      // Remount on a new set: the list keeps its own per-review edits.
      key={section.reviews.map((r) => r.node_exec_id).join(",")}
      reviews={section.reviews}
      onReviewComplete={live.onReviewed}
      emptyMessage="Nothing is waiting on you."
    />
  );
}
