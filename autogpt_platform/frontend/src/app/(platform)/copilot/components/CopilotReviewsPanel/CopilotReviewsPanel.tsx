"use client";

import { useState } from "react";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ClockIcon, XIcon } from "@phosphor-icons/react";
import { cn } from "@/lib/utils";

/** Must match COPILOT_SESSION_PREFIX in backend/copilot/tools/run_block.py */
const COPILOT_SESSION_PREFIX = "copilot-session-";

interface CopilotReviewsPanelProps {
  sessionId: string;
  className?: string;
}

export function CopilotReviewsPanel({
  sessionId,
  className,
}: CopilotReviewsPanelProps) {
  const [isOpen, setIsOpen] = useState(false);

  const syntheticGraphExecId = `${COPILOT_SESSION_PREFIX}${sessionId}`;

  const { pendingReviews, isLoading, refetch } = usePendingReviewsForExecution(
    syntheticGraphExecId,
    {
      enabled: true,
      refetchInterval: 2000,
    },
  );

  if (!isLoading && pendingReviews.length === 0) {
    return null;
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
                onReviewComplete={() => {
                  refetch();
                  setIsOpen(false);
                }}
                emptyMessage="No pending reviews"
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}
