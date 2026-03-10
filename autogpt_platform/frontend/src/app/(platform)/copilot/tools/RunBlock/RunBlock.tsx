"use client";

import type { ToolUIPart } from "ai";
import { useCallback } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { usePendingReviewsForExecution } from "@/hooks/usePendingReviews";
import { okData } from "@/app/api/helpers";
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

  // Fetch real pending reviews from API (survives page refresh)
  const { pendingReviews, refetch } = usePendingReviewsForExecution(
    reviewOutput?.graph_exec_id ?? "",
    { enabled: !!reviewOutput?.graph_exec_id, refetchInterval: 2000 },
  );
  // Filter to only the review for this specific block execution
  const reviewsForThisBlock = reviewOutput
    ? pendingReviews.filter((r) => r.node_exec_id === reviewOutput.review_id)
    : [];
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

  // After approval, check if all reviews for this session are done before
  // telling the LLM to continue. This prevents a separate chat turn per review
  // when multiple blocks are pending approval simultaneously.
  const handleReviewComplete = useCallback(async () => {
    if (!reviewOutput) return;

    // Brief delay for the server to propagate the approval
    await new Promise((resolve) => setTimeout(resolve, 500));
    const result = await refetch();
    const remaining = okData(result.data) || [];

    if (remaining.length > 0) {
      // More reviews still pending — don't trigger the LLM yet
      return;
    }

    // All reviews approved — send a single message for the LLM to continue
    onSend(
      `All pending reviews have been approved. ` +
        `Please call continue_run_block for each review_id from the previous run_block results to execute them.`,
    );
  }, [reviewOutput, refetch, onSend]);

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

      {reviewsForThisBlock.length > 0 && (
        <div className="mt-2">
          <PendingReviewsList
            reviews={reviewsForThisBlock}
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
