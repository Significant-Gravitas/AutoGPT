"use client";

import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import { BlockDetailsCard } from "./components/BlockDetailsCard/BlockDetailsCard";
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
  isRunBlockSetupRequirementsOutput,
  ToolIcon,
} from "./helpers";

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
    (isRunBlockBlockOutput(output) ||
      isRunBlockDetailsOutput(output) ||
      isRunBlockErrorOutput(output));

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
