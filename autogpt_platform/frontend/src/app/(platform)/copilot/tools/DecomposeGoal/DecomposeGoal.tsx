"use client";

import type { ToolUIPart } from "ai";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import {
  ContentGrid,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { ToolErrorCard } from "../../components/ToolErrorCard/ToolErrorCard";
import { StepItem } from "./components/StepItem";
import {
  AccordionIcon,
  getAnimationText,
  getDecomposeGoalOutput,
  isDecompositionOutput,
  isErrorOutput,
  ToolIcon,
} from "./helpers";

interface Props {
  part: ToolUIPart;
}

export function DecomposeGoalTool({ part }: Props) {
  const text = getAnimationText(part);
  const { onSend } = useCopilotChatActions();

  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getDecomposeGoalOutput(part);
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));
  const isPending = !output && !isError;

  return (
    <div className="py-2">
      {isPending && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <ToolIcon isStreaming={isStreaming} isError={isError} />
          <MorphingTextAnimation
            text={text}
            className={isError ? "text-red-500" : undefined}
          />
        </div>
      )}

      {isError && (
        <ToolErrorCard
          message={
            output && isErrorOutput(output) ? (output.message ?? "") : ""
          }
          fallbackMessage="Failed to analyze the goal. Please try again."
          actions={[
            {
              label: "Try again",
              onClick: () => onSend("Please try decomposing the goal again."),
            },
          ]}
        />
      )}

      {output && isDecompositionOutput(output) && (
        <ToolAccordion
          icon={<AccordionIcon />}
          title={`Build Plan — ${output.step_count} steps`}
          description={output.goal}
          defaultExpanded
        >
          <ContentGrid>
            <ContentMessage>{output.message}</ContentMessage>

            <div className="rounded-lg border border-border bg-card p-3">
              <div className="space-y-0.5">
                {output.steps.map((step, i) => (
                  <StepItem
                    key={step.step_id}
                    index={i}
                    description={step.description}
                    blockName={step.block_name}
                    status={step.status ?? "pending"}
                  />
                ))}
              </div>
            </div>
          </ContentGrid>
        </ToolAccordion>
      )}
    </div>
  );
}
