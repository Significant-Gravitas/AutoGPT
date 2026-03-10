"use client";

import { useState } from "react";
import { usePostV2ProcessReviewAction } from "@/app/api/__generated__/endpoints/executions/executions";
import { Button } from "@/components/atoms/Button/Button";
import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  ContentCodeBlock,
  ContentGrid,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import { formatMaybeJson } from "../../helpers";
import type { ReviewRequiredResponse } from "../../helpers";

interface Props {
  output: ReviewRequiredResponse;
}

export function ReviewRequiredCard({ output }: Props) {
  const [actionTaken, setActionTaken] = useState<
    "approved" | "rejected" | null
  >(null);
  const [pendingAction, setPendingAction] = useState<
    "approve" | "reject" | null
  >(null);
  const { toast } = useToast();

  const reviewAction = usePostV2ProcessReviewAction({
    mutation: {
      onSuccess: (res) => {
        if (res.status !== 200) {
          toast({
            title: "Failed to process review",
            variant: "destructive",
          });
          setPendingAction(null);
          return;
        }
        const result = res.data;
        if (result.approved_count > 0) {
          setActionTaken("approved");
        } else if (result.rejected_count > 0) {
          setActionTaken("rejected");
        }
        setPendingAction(null);
      },
      onError: (error: Error) => {
        setPendingAction(null);
        toast({
          title: "Failed to process review",
          description: error.message,
          variant: "destructive",
        });
      },
    },
  });

  function handleAction(approved: boolean) {
    setPendingAction(approved ? "approve" : "reject");
    reviewAction.mutate({
      data: {
        reviews: [
          {
            node_exec_id: output.review_id,
            approved,
            reviewed_data: output.input_data,
            auto_approve_future: approved,
          },
        ],
      },
    });
  }

  return (
    <ContentGrid>
      <ContentMessage>{output.message}</ContentMessage>
      {Object.keys(output.input_data).length > 0 && (
        <ContentCodeBlock>
          {formatMaybeJson(output.input_data)}
        </ContentCodeBlock>
      )}
      {actionTaken ? (
        <ContentMessage className="font-medium">
          {actionTaken === "approved"
            ? "Approved — the block will execute on next attempt."
            : "Rejected — the block will not execute."}
        </ContentMessage>
      ) : (
        <div className="flex gap-2 pt-1">
          <Button
            onClick={() => handleAction(true)}
            disabled={reviewAction.isPending}
            loading={pendingAction === "approve"}
            variant="primary"
            size="small"
            className="rounded-full px-4"
          >
            Approve
          </Button>
          <Button
            onClick={() => handleAction(false)}
            disabled={reviewAction.isPending}
            loading={pendingAction === "reject"}
            variant="destructive"
            size="small"
            className="rounded-full bg-red-600 px-4"
          >
            Reject
          </Button>
        </div>
      )}
    </ContentGrid>
  );
}
