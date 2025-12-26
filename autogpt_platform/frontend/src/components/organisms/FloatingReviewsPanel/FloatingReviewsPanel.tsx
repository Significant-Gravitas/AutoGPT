import { useState, useEffect } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { Button } from "@/components/atoms/Button/Button";
import { ClockIcon, XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";
import { Text } from "@/components/atoms/Text/Text";
import { useGetV1GetExecutionDetails } from "@/app/api/__generated__/endpoints/graphs/graphs";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { okData } from "@/app/api/helpers";
import { useGraphStore } from "@/app/(platform)/build/stores/graphStore";
import { useShallow } from "zustand/react/shallow";

interface FloatingReviewsPanelProps {
  executionId?: string;
  graphId?: string;
  className?: string;
}

export function FloatingReviewsPanel({
  executionId,
  graphId,
  className,
}: FloatingReviewsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const { data: executionDetails } = useGetV1GetExecutionDetails(
    graphId || "",
    executionId || "",
    {
      query: {
        enabled: !!(graphId && executionId),
        select: okData,
      },
    },
  );

  // Get graph execution status from the store (updated via WebSocket)
  const graphExecutionStatus = useGraphStore(
    useShallow((state) => state.graphExecutionStatus),
  );

  const { pendingReviews, isLoading, refetch } = usePendingReviewsForExecution(
    executionId || "",
  );

  useEffect(() => {
    if (executionId) {
      refetch();
    }
  }, [executionDetails?.status, executionId, refetch]);

  // Refetch when graph execution status changes to REVIEW
  useEffect(() => {
    if (graphExecutionStatus === AgentExecutionStatus.REVIEW && executionId) {
      refetch();
    }
  }, [graphExecutionStatus, executionId, refetch]);

  if (
    !executionId ||
    (!isLoading &&
      pendingReviews.length === 0 &&
      executionDetails?.status !== AgentExecutionStatus.REVIEW)
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
        <div className="relative flex max-h-[80vh] max-w-2xl flex-col overflow-hidden rounded-lg shadow-2xl">
          <Button
            onClick={() => setIsOpen(false)}
            variant="icon"
            size="icon"
            className="absolute right-4 top-4 z-10"
          >
            <XIcon size={16} />
          </Button>

          <div className="flex-1 overflow-y-auto">
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
