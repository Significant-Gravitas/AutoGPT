"use client";

import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ToolUIPart } from "ai";
import { useCallback, useMemo } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { ReviewStatus } from "@/app/api/__generated__/models/reviewStatus";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { BlockDetailsCard } from "./components/BlockDetailsCard/BlockDetailsCard";
import { BlockInputCard } from "./components/BlockInputCard/BlockInputCard";
import { BlockOutputCard } from "./components/BlockOutputCard/BlockOutputCard";
import { ErrorCard } from "./components/ErrorCard/ErrorCard";
import { SetupRequirementsCard } from "./components/SetupRequirementsCard/SetupRequirementsCard";
import {
  getAccordionMeta,
  getAnimationText,
  getRunBlockToolOutput,
  isRunBlockBlockOutput,
  isRunBlockDetailsOutput,
  isRunBlockErrorOutput,
  isRunBlockReviewRequiredOutput,
  isRunBlockSetupRequirementsOutput,
  ToolIcon,
} from "./helpers";
import type { ReviewRequiredResponse, RunBlockInput } from "./helpers";

export interface RunBlockToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: RunBlockToolPart;
}

export function RunBlockTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const { onSend } = useCopilotChatActions();

  const output = getRunBlockToolOutput(part);
  const inputData = (part.input as RunBlockInput | undefined)?.input_data;
  const hasInputData = inputData != null && Object.keys(inputData).length > 0;
  const isError =
    part.state === "output-error" ||
    (!!output && isRunBlockErrorOutput(output));
  const reviewOutput =
    output && isRunBlockReviewRequiredOutput(output)
      ? (output as ReviewRequiredResponse)
      : null;

  // Check if the review is still pending (survives page refresh)
  const { pendingReviews } = usePendingReviewsForExecution(
    reviewOutput?.session_id ?? "",
    { enabled: !!reviewOutput?.session_id },
  );
  const isReviewStillPending =
    !!reviewOutput &&
    pendingReviews.some((r) => r.node_exec_id === reviewOutput.review_id);
  const setupRequirementsOutput =
    part.state === "output-available" &&
    output &&
    isRunBlockSetupRequirementsOutput(output)
      ? output
      : null;

  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    !setupRequirementsOutput &&
    !isRunBlockReviewRequiredOutput(output) &&
    (isRunBlockBlockOutput(output) ||
      isRunBlockDetailsOutput(output) ||
      isRunBlockErrorOutput(output));

  // Convert ReviewRequiredResponse to PendingHumanReviewModel for reuse
  const reviewAsPendingReview = useMemo((): PendingHumanReviewModel[] => {
    if (!reviewOutput) return [];
    return [
      {
        node_exec_id: reviewOutput.review_id,
        node_id: reviewOutput.block_id,
        user_id: "",
        graph_exec_id: reviewOutput.session_id ?? "",
        graph_id: "",
        graph_version: 0,
        payload: reviewOutput.input_data,
        instructions: reviewOutput.block_name,
        editable: true,
        status: ReviewStatus.WAITING,
        created_at: new Date(),
      },
    ];
  }, [reviewOutput]);

  // After approval, automatically tell the LLM to continue with the review_id
  const handleReviewComplete = useCallback(() => {
    if (!reviewOutput) return;
    onSend(
      `The review for "${reviewOutput.block_name}" has been approved. ` +
        `Please call continue_run_block with review_id="${reviewOutput.review_id}" to execute it.`,
    );
  }, [reviewOutput, onSend]);

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {setupRequirementsOutput && (
        <div className="mt-2">
          <SetupRequirementsCard output={setupRequirementsOutput} />
        </div>
      )}

      {isReviewStillPending && reviewAsPendingReview.length > 0 && (
        <div className="mt-2">
          <PendingReviewsList
            reviews={reviewAsPendingReview}
            onReviewComplete={handleReviewComplete}
          />
        </div>
      )}

      {hasExpandableContent && output && (
        <ToolAccordion {...getAccordionMeta(output)}>
          {hasInputData && <BlockInputCard inputData={inputData} />}

          {isRunBlockBlockOutput(output) && <BlockOutputCard output={output} />}

          {isRunBlockDetailsOutput(output) && (
            <BlockDetailsCard output={output} />
          )}

          {isRunBlockErrorOutput(output) && <ErrorCard output={output} />}
        </ToolAccordion>
      )}
    </div>
  );
}
