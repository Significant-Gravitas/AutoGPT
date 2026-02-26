"use client";

import {
  ContentCard,
  ContentCardTitle,
  ContentCodeBlock,
  ContentGrid,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import type { MCPToolOutputResult } from "../../helpers";

interface Props {
  output: MCPToolOutputResult;
}

function formatResult(value: unknown): string {
  if (value === null || value === undefined) return "(no result)";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function MCPToolOutputCard({ output }: Props) {
  const resultText = formatResult(output.result);
  const isJson =
    output.result !== null &&
    output.result !== undefined &&
    typeof output.result !== "string";

  return (
    <ContentGrid>
      <ContentMessage>{output.message}</ContentMessage>

      <ContentCard>
        <ContentCardTitle className="text-xs">
          {output.tool_name}
        </ContentCardTitle>
        {isJson ? (
          <ContentCodeBlock className="mt-2">{resultText}</ContentCodeBlock>
        ) : (
          <p className="mt-2 whitespace-pre-wrap break-words text-sm text-zinc-800">
            {resultText}
          </p>
        )}
      </ContentCard>
    </ContentGrid>
  );
}
