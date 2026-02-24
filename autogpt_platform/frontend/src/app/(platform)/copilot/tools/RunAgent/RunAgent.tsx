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
import { MiniGame } from "../../components/MiniGame/MiniGame";
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
  const isOutputAvailable = part.state === "output-available" && !!output;

  const setupRequirementsOutput =
    isOutputAvailable && isRunAgentSetupRequirementsOutput(output)
      ? output
      : null;

  const agentDetailsOutput =
    isOutputAvailable && isRunAgentAgentDetailsOutput(output) ? output : null;

  const needLoginOutput =
    isOutputAvailable && isRunAgentNeedLoginOutput(output) ? output : null;

  const hasExpandableContent =
    isOutputAvailable &&
    !setupRequirementsOutput &&
    !agentDetailsOutput &&
    !needLoginOutput &&
    (isRunAgentExecutionStartedOutput(output) || isRunAgentErrorOutput(output));

  return (
    <div className="py-2">
      {/* Only show loading text when NOT showing accordion or other content */}
      {!isStreaming &&
        !setupRequirementsOutput &&
        !agentDetailsOutput &&
        !needLoginOutput &&
        !hasExpandableContent && (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ToolIcon isStreaming={isStreaming} isError={isError} />
            <MorphingTextAnimation
              text={text}
              className={isError ? "text-red-500" : undefined}
            />
          </div>
        )}

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

      {setupRequirementsOutput && (
        <div className="mt-2">
          <SetupRequirementsCard output={setupRequirementsOutput} />
        </div>
      )}

      {agentDetailsOutput && (
        <div className="mt-2">
          <AgentDetailsCard output={agentDetailsOutput} />
        </div>
      )}

      {needLoginOutput && (
        <div className="mt-2">
          <ContentMessage>{needLoginOutput.message}</ContentMessage>
        </div>
      )}

      {hasExpandableContent && output && (
        <ToolAccordion {...getAccordionMeta(output)}>
          {isRunAgentExecutionStartedOutput(output) && (
            <ExecutionStartedCard output={output} />
          )}

          {isRunAgentErrorOutput(output) && <ErrorCard output={output} />}
        </ToolAccordion>
      )}
    </div>
  );
}
