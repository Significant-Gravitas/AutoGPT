"use client";

import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { OrbitLoader } from "../../components/OrbitLoader/OrbitLoader";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ContentGrid,
  ContentHint,
  ContentMessage,
} from "../../components/ToolAccordion/AccordionContent";
import { MiniGame } from "../CreateAgent/components/MiniGame/MiniGame";
import {
  getAccordionMeta,
  getAnimationText,
  getRunAgentToolOutput,
  isRunAgentAgentDetailsOutput,
  isRunAgentErrorOutput,
  isRunAgentExecutionStartedOutput,
  isRunAgentNeedLoginOutput,
  isRunAgentSetupRequirementsOutput,
  ToolIcon,
} from "./helpers";
import { ExecutionStartedCard } from "./components/ExecutionStartedCard/ExecutionStartedCard";
import { AgentDetailsCard } from "./components/AgentDetailsCard/AgentDetailsCard";
import { SetupRequirementsCard } from "./components/SetupRequirementsCard/SetupRequirementsCard";
import { ErrorCard } from "./components/ErrorCard/ErrorCard";

export interface RunAgentToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: RunAgentToolPart;
}

export function RunAgentTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getRunAgentToolOutput(part);
  const isError =
    part.state === "output-error" ||
    (!!output && isRunAgentErrorOutput(output));
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (isRunAgentExecutionStartedOutput(output) ||
      isRunAgentAgentDetailsOutput(output) ||
      isRunAgentSetupRequirementsOutput(output) ||
      isRunAgentNeedLoginOutput(output) ||
      isRunAgentErrorOutput(output));

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {isStreaming && !output && (
        <ToolAccordion
          icon={<OrbitLoader size={32} />}
          title="Running agent, this may take a few minutes. Play while you wait."
          expanded={true}
        >
          <ContentGrid>
            <MiniGame />
            <ContentHint>
              This could take a few minutes — play while you wait!
            </ContentHint>
          </ContentGrid>
        </ToolAccordion>
      )}

      {hasExpandableContent && output && (
        <ToolAccordion {...getAccordionMeta(output)}>
          {isRunAgentExecutionStartedOutput(output) && (
            <ExecutionStartedCard output={output} />
          )}

          {isRunAgentAgentDetailsOutput(output) && (
            <AgentDetailsCard output={output} />
          )}

          {isRunAgentSetupRequirementsOutput(output) && (
            <SetupRequirementsCard output={output} />
          )}

          {isRunAgentNeedLoginOutput(output) && (
            <ContentMessage>{output.message}</ContentMessage>
          )}

          {isRunAgentErrorOutput(output) && <ErrorCard output={output} />}
        </ToolAccordion>
      )}
    </div>
  );
}
