"use client";

import { MorphingTextAnimation } from "../../components/MorphingTextAnimation/MorphingTextAnimation";
import { ToolAccordion } from "../../components/ToolAccordion/ToolAccordion";
import {
  ContentCard,
  ContentCardDescription,
  ContentCardTitle,
} from "../../components/ToolAccordion/AccordionContent";
import type { BlockListResponse } from "@/app/api/__generated__/models/blockListResponse";
import type { BlockInfoSummary } from "@/app/api/__generated__/models/blockInfoSummary";
import { ToolUIPart } from "ai";
import { HorizontalScroll } from "@/app/(platform)/build/components/NewControlPanel/NewBlockMenu/HorizontalScroll";
import {
  AccordionIcon,
  getAnimationText,
  parseOutput,
  ToolIcon,
} from "./helpers";

export interface FindBlockInput {
  query: string;
}

export type FindBlockOutput = BlockListResponse;

export interface FindBlockToolPart {
  type: string;
  toolName?: string;
  toolCallId: string;
  state: ToolUIPart["state"];
  input?: FindBlockInput | unknown;
  output?: string | FindBlockOutput | unknown;
  title?: string;
}

interface Props {
  part: FindBlockToolPart;
}

function BlockCard({ block }: { block: BlockInfoSummary }) {
  return (
    <ContentCard className="w-48 shrink-0">
      <ContentCardTitle>{block.name}</ContentCardTitle>
      <ContentCardDescription className="mt-1 line-clamp-2">
        {block.description}
      </ContentCardDescription>
    </ContentCard>
  );
}

export function FindBlocksTool({ part }: Props) {
  const text = getAnimationText(part);
  const isStreaming =
    part.state === "input-streaming" || part.state === "input-available";
  const isError = part.state === "output-error";

  const parsed =
    part.state === "output-available" ? parseOutput(part.output) : null;
  const hasBlocks = !!parsed && parsed.blocks.length > 0;

  const query = (part.input as FindBlockInput | undefined)?.query?.trim();
  const accordionDescription = parsed
    ? `Found ${parsed.count} block${parsed.count === 1 ? "" : "s"}${query ? ` for "${query}"` : ""}`
    : undefined;

  return (
    <div className="py-2">
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <ToolIcon isStreaming={isStreaming} isError={isError} />
        <MorphingTextAnimation
          text={text}
          className={isError ? "text-red-500" : undefined}
        />
      </div>

      {hasBlocks && parsed && (
        <ToolAccordion
          icon={<AccordionIcon />}
          title="Block results"
          description={accordionDescription}
        >
          <HorizontalScroll dependencyList={[parsed.blocks.length]}>
            {parsed.blocks.map((block) => (
              <BlockCard key={block.id} block={block} />
            ))}
          </HorizontalScroll>
        </ToolAccordion>
      )}
    </div>
  );
}
