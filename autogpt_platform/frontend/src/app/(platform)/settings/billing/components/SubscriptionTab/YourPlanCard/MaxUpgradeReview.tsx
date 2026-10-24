import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { MaxUpgradeSummary } from "./MaxUpgradeSummary";
import type { useTierUpgradePreview } from "./useTierUpgradePreview";

interface Props {
  review: ReturnType<typeof useTierUpgradePreview>;
}

export function MaxUpgradeReview({ review }: Props) {
  return (
    <div aria-live="polite">
      {review.isLoading && (
        <div className="space-y-3">
          <Text variant="small" tone="secondary">
            Reviewing your charge…
          </Text>
          <Skeleton className="h-48 rounded-2xl" />
        </div>
      )}
      {review.error && (
        <div
          role="alert"
          // TODO(tokens): no exact role (border-orange-200 #FFC88A, bg-orange-50 #FFF3E6)
          className="rounded-xl border border-orange-200 bg-orange-50 p-4"
        >
          <Text variant="small" className="mb-3 leading-6">
            {review.error}
          </Text>
          <Button variant="secondary" size="small" onClick={review.retry}>
            Retry review
          </Button>
        </div>
      )}
      {review.preview && <MaxUpgradeSummary preview={review.preview} />}
    </div>
  );
}
