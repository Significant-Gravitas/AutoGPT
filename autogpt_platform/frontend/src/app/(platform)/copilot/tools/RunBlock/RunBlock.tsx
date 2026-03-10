"use client";

import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import type { ToolUIPart } from "ai";
import { useMemo } from "react";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { PendingReviewsList } from "@/components/organisms/PendingReviewsList/PendingReviewsList";
import { ReviewStatus } from "@/app/api/__generated__/models/reviewStatus";
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
import type { RunBlockInput } from "./helpers";

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

  const output = getRunBlockToolOutput(part);
  const inputData = (part.input as RunBlockInput | undefined)?.input_data;
  const hasInputData = inputData != null && Object.keys(inputData).length > 0;
  const isError =
    part.state === "output-error" ||
    (!!output && isRunBlockErrorOutput(output));
  const isReviewRequired = !!output && isRunBlockReviewRequiredOutput(output);
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
    if (!output || !isRunBlockReviewRequiredOutput(output)) return [];
    return [
      {
        node_exec_id: output.review_id,
        node_id: output.block_id,
        user_id: "",
        graph_exec_id: output.session_id ?? "",
        graph_id: "",
        graph_version: 0,
        payload: output.input_data,
        instructions: output.block_name,
        editable: false,
        status: ReviewStatus.WAITING,
        created_at: new Date(),
      },
    ];
  }, [output]);

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={
            isError
              ? "text-red-500"
              : isReviewRequired
                ? "text-amber-500"
                : undefined
          }
        />
      </div>

      {setupRequirementsOutput && (
        <div className="mt-2">
          <SetupRequirementsCard output={setupRequirementsOutput} />
        </div>
      )}

      {isReviewRequired && reviewAsPendingReview.length > 0 && (
        <div className="mt-2">
          <PendingReviewsList reviews={reviewAsPendingReview} />
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
