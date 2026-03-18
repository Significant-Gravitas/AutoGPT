"use client";

import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
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

  // Review UI is rendered at the chat level by CopilotPendingReviews,
  // not inside each tool card. This matches the non-copilot flow where
  // a single PendingReviewsList shows all reviews grouped together.

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
