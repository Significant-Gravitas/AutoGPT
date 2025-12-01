import { useState, useEffect } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { Button } from "@/components/atoms/Button/Button";
import { ClockIcon, XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";

interface FloatingReviewsPanelProps {
  executionId?: string;
  className?: string;
}

export function FloatingReviewsPanel({
  executionId,
  className,
}: FloatingReviewsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const executionStatus = useGraphStore((state) => state.graphExecutionStatus);

  const { pendingReviews, isLoading, refetch } = usePendingReviewsForExecution(
    executionId || "",
  );

  useEffect(() => {
    if (executionStatus === AgentExecutionStatus.REVIEW && executionId) {
      refetch();
    }
  }, [executionStatus, executionId, refetch]);

  if (
    !executionId ||
    (!isLoading && pendingReviews.length === 0) ||
    executionStatus !== AgentExecutionStatus.REVIEW
  ) {
    return null;
  }

  function handleReviewComplete() {
    refetch();
    setIsOpen(false);
  }

  return (
    <div className={cn("fixed bottom-20 right-4 z-50", className)}>
      {!isOpen && pendingReviews.length > 0 && (
        <Button
          onClick={() => setIsOpen(true)}
          size="large"
          variant="primary"
          leftIcon={<ClockIcon size={20} />}
        >
          {pendingReviews.length} Review
          {pendingReviews.length !== 1 ? "s" : ""} Pending
        </Button>
      )}

      {isOpen && (
        <div className="flex max-h-[80vh] max-w-2xl flex-col overflow-hidden rounded-lg border bg-white shadow-2xl">
          <div className="flex items-center justify-between border-b bg-gray-50 p-4">
            <div className="flex items-center gap-2">
              <ClockIcon size={20} className="text-orange-600" />
              <Text variant="h4">Pending Reviews</Text>
            </div>
            <Button onClick={() => setIsOpen(false)} variant="icon" size="icon">
              <XIcon size={16} />
            </Button>
          </div>

          <div className="flex-1 overflow-y-auto p-4">
            {isLoading ? (
              <div className="py-8 text-center">
                <Text variant="body" className="text-muted-foreground">
                  Loading reviews...
                </Text>
              </div>
            ) : (
              <PendingReviewsList
                reviews={pendingReviews}
                onReviewComplete={handleReviewComplete}
                emptyMessage="No pending reviews"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
