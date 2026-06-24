"use client";

import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { ContentMessage } from "../../components/ToolAccordion/AccordionContent";
import {
  getAccordionMeta,
  getAnimationText,
  getRunAgentToolOutput,
  getStreamingLoadingText,
  isRunAgentAgentDetailsOutput,
  isRunAgentAgentOutputResponse,
  isRunAgentErrorOutput,
  isRunAgentExecutionStartedOutput,
  isRunAgentNeedLoginOutput,
  isRunAgentSetupRequirementsOutput,
  ToolIcon,
} from "./helpers";
import { ExecutionStartedCard } from "./components/ExecutionStartedCard/ExecutionStartedCard";
import { AgentDetailsCard } from "./components/AgentDetailsCard/AgentDetailsCard";
import { SetupRequirementsCard } from "../../components/SetupRequirementsCard/SetupRequirementsCard";
import { ErrorCard } from "./components/ErrorCard/ErrorCard";
import {
  isUnparseableJsonOutput,
  reportCorruptedToolOutput,
} from "../../helpers/toolOutput";

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
  const isCorrupted =
    part.state === "output-available" &&
    !output &&
    isUnparseableJsonOutput(part.output);
  if (isCorrupted) reportCorruptedToolOutput(part.toolCallId, part.type);
  const isError =
    part.state === "output-error" ||
    isCorrupted ||
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

  const agentOutputResponse =
    isOutputAvailable && isRunAgentAgentOutputResponse(output) ? output : null;

  const hasExpandableContent =
    isOutputAvailable &&
    !setupRequirementsOutput &&
    !agentDetailsOutput &&
    !needLoginOutput &&
    (isRunAgentExecutionStartedOutput(output) ||
      isRunAgentAgentOutputResponse(output) ||
      isRunAgentErrorOutput(output));

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
              text={isCorrupted ? "Agent result could not be displayed" : text}
              className={isError ? "text-red-500" : undefined}
            />
          </div>
        )}

      {isCorrupted && (
        <p className="mt-1 text-sm text-red-500">
          The result data arrived corrupted, so any sign-in or setup card it
          contained can&apos;t be shown. Ask AutoPilot to retry this step.
        </p>
      )}

      {isStreaming && !output && (
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <ToolIcon isStreaming isError={isError} />
          <MorphingTextAnimation text={getStreamingLoadingText(part)} />
        </div>
      )}

      {setupRequirementsOutput && (
        <div className="mt-2">
          <SetupRequirementsCard
            output={setupRequirementsOutput}
            inputsMode="preview"
          />
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

          {agentOutputResponse && (
            <ExecutionStartedCard
              output={{
                message: agentOutputResponse.message,
                execution_id: agentOutputResponse.execution?.execution_id ?? "",
                graph_id: agentOutputResponse.agent_id,
                graph_name: agentOutputResponse.agent_name,
                library_agent_link:
                  agentOutputResponse.library_agent_link ?? undefined,
                // Propagate the real terminal status (COMPLETED / FAILED /
                // STOPPED …) so the card title matches what happened.
                // Defaults to the "started" label when backend omits status.
                status: agentOutputResponse.execution?.status ?? "COMPLETED",
              }}
            />
          )}

          {isRunAgentErrorOutput(output) && <ErrorCard output={output} />}
        </ToolAccordion>
      )}
    </div>
  );
}
