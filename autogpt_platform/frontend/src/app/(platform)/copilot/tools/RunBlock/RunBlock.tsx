"use client";

import type { ToolUIPart } from "ai";
import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ContentGrid,
  ContentHint,
} from "../../components/ToolAccordion/AccordionContent";
import { MiniGame } from "../CreateAgent/components/MiniGame/MiniGame";
import { BlockOutputCard } from "./components/BlockOutputCard/BlockOutputCard";
import { ErrorCard } from "./components/ErrorCard/ErrorCard";
import { SetupRequirementsCard } from "./components/SetupRequirementsCard/SetupRequirementsCard";
import {
  AccordionIcon,
  getAccordionMeta,
  getAnimationText,
  getRunBlockToolOutput,
  isRunBlockBlockOutput,
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
  const hasExpandableContent =
    part.state === "output-available" &&
    !!output &&
    (isRunBlockBlockOutput(output) ||
      isRunBlockSetupRequirementsOutput(output) ||
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

      {isStreaming && !output && (
        <ToolAccordion
          icon={<AccordionIcon />}
          title="Running block, this may take a moment..."
          expanded
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
          {isRunBlockBlockOutput(output) && <BlockOutputCard output={output} />}

          {isRunBlockSetupRequirementsOutput(output) && (
            <SetupRequirementsCard output={output} />
          )}

          {isRunBlockErrorOutput(output) && <ErrorCard output={output} />}
        </ToolAccordion>
      )}
    </div>
  );
}
