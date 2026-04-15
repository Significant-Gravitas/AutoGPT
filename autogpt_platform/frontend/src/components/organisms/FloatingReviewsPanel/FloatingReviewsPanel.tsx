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
        // Poll while execution is in progress to detect status changes
        refetchInterval: (q) => {
          // Note: refetchInterval callback receives raw data before select transform
          const rawData = q.state.data as
            | { status: number; data?: { status?: string } }
            | undefined;
          if (rawData?.status !== 200) return false;

          const status = rawData?.data?.status;
          if (!status) return false;

          // Poll every 2 seconds while running or in review
          if (
            status === AgentExecutionStatus.RUNNING ||
            status === AgentExecutionStatus.QUEUED ||
            status === AgentExecutionStatus.INCOMPLETE ||
            status === AgentExecutionStatus.REVIEW
          ) {
            return 2000;
          }
          return false;
        },
        refetchIntervalInBackground: true,
      },
    },
  );

  // Get graph execution status from the store (updated via WebSocket)
  const graphExecutionStatus = useGraphStore(
    useShallow((state) => state.graphExecutionStatus),
  );

  // Determine if we should poll for pending reviews
  const isInReviewStatus =
    executionDetails?.status === AgentExecutionStatus.REVIEW ||
    graphExecutionStatus === AgentExecutionStatus.REVIEW;

  const { pendingReviews, isLoading, refetch } = usePendingReviewsForExecution(
    executionId || "",
    {
      enabled: !!executionId,
      // Poll every 2 seconds when in REVIEW status to catch new reviews
      refetchInterval: isInReviewStatus ? 2000 : false,
    },
  );

  // Refetch pending reviews when execution status changes
  useEffect(() => {
    if (executionId && executionDetails?.status) {
      refetch();
    }
  }, [executionDetails?.status, executionId, refetch]);

  // Hide panel if:
  // 1. No execution ID
  // 2. No pending reviews and not in REVIEW status
  // 3. Execution is RUNNING or QUEUED (hasn't paused for review yet)
  if (!executionId) {
    return null;
  }

  if (
    !isLoading &&
    pendingReviews.length === 0 &&
    executionDetails?.status !== AgentExecutionStatus.REVIEW
  ) {
    return null;
  }

  // Don't show panel while execution is still running/queued (not paused for review)
  if (
    executionDetails?.status === AgentExecutionStatus.RUNNING ||
    executionDetails?.status === AgentExecutionStatus.QUEUED
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
