"use client";

import { PlugsConnectedIcon } from "@phosphor-icons/react";
import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { SetupRequirementsCard } from "../RunBlock/components/SetupRequirementsCard/SetupRequirementsCard";
import { MCPToolOutputCard } from "./components/MCPToolOutputCard/MCPToolOutputCard";
import {
  ToolIcon,
  getAnimationText,
  getRunMCPToolOutput,
  isErrorOutput,
  isMCPToolOutput,
  isSetupRequirementsOutput,
  serverHost,
} from "./helpers";

export interface RunMCPToolPart {
  type: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
}

interface Props {
  part: RunMCPToolPart;
}

export function RunMCPToolComponent({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";

  const output = getRunMCPToolOutput(part);
  const isError =
    part.state === "output-error" || (!!output && isErrorOutput(output));

  const setupRequirementsOutput =
    part.state === "output-available" &&
    output &&
    isSetupRequirementsOutput(output)
      ? output
      : null;

  const mcpToolOutput =
    part.state === "output-available" && output && isMCPToolOutput(output)
      ? output
      : null;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {/* Credential setup — same card as used in RunBlock / graph builder */}
      {setupRequirementsOutput && (
        <div className="mt-2">
          <SetupRequirementsCard output={setupRequirementsOutput} />
        </div>
      )}

      {/* Tool execution result */}
      {mcpToolOutput && (
        <ToolAccordion
          icon={<PlugsConnectedIcon size={32} weight="light" />}
          title={mcpToolOutput.tool_name}
          description={`from ${serverHost(mcpToolOutput.server_url)}`}
        >
          <MCPToolOutputCard output={mcpToolOutput} />
        </ToolAccordion>
      )}

      {/* Discovery output: consumed by the agent automatically — no user-facing UI */}
    </div>
  );
}
