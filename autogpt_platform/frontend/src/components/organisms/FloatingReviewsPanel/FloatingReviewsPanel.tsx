import { useState, useEffect } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { Button } from "@/components/atoms/Button/Button";
import { ClockIcon, XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

interface FloatingReviewsPanelProps {
  executionId?: string;
  graphId?: string;
  executionStatus?: AgentExecutionStatus;
  className?: string;
}

export function FloatingReviewsPanel({
  executionId,
  executionStatus,
  className,
}: FloatingReviewsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const { pendingReviews, isLoading, refetch } = usePendingReviewsForExecution(
    executionId || "",
  );

  // Refetch pending reviews when execution status changes to REVIEW
  useEffect(() => {
    if (executionStatus === AgentExecutionStatus.REVIEW && executionId) {
      refetch();
    }
  }, [executionStatus, executionId, refetch]);

  // Don't show anything if there's no execution ID, no pending reviews, or execution status is not REVIEW
  if (
    !executionId ||
    (!isLoading && pendingReviews.length === 0) ||
    executionStatus !== AgentExecutionStatus.REVIEW
  ) {
    return null;
  }

  function handleReviewComplete() {
    refetch();
    // Close panel and let render logic handle visibility
    setIsOpen(false);
  }

  return (
    <div className={cn("fixed bottom-20 right-4 z-50", className)}>
      {/* Trigger Button */}
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

      {/* Reviews Panel */}
      {isOpen && (
        <div className="flex max-h-[80vh] max-w-2xl flex-col overflow-hidden rounded-lg border bg-white shadow-2xl">
          {/* Header */}
          <div className="flex items-center justify-between border-b bg-gray-50 p-4">
            <div className="flex items-center gap-2">
              <ClockIcon size={20} className="text-orange-600" />
              <Text variant="h4">Pending Reviews</Text>
            </div>
            <Button onClick={() => setIsOpen(false)} variant="icon" size="icon">
              <XIcon size={16} />
            </Button>
          </div>

          {/* Content */}
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
