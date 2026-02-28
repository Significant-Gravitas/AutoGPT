"use client";

import {
  ContentCard,
  ContentCardTitle,
  ContentCodeBlock,
  ContentGrid,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import type { MCPToolOutputResponse } from "../../helpers";

interface Props {
  output: MCPToolOutputResponse;
}

interface ImageResult {
  type: "image";
  data: string;
  mimeType: string;
}

function isImageResult(value: unknown): value is ImageResult {
  return (
    typeof value === "object" &&
    value !== null &&
    (value as Record<string, unknown>).type === "image" &&
    typeof (value as Record<string, unknown>).data === "string" &&
    typeof (value as Record<string, unknown>).mimeType === "string"
  );
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
  const result = output.result;

  const isImage = isImageResult(result);
  const resultText = isImage ? "" : formatResult(result);
  const isJson =
    !isImage &&
    result !== null &&
    result !== undefined &&
    typeof result !== "string";

  return (
    <ContentGrid>
      <ContentMessage>{output.message}</ContentMessage>

      <ContentCard>
        <ContentCardTitle className="text-xs">
          {output.tool_name}
        </ContentCardTitle>

        {isImage ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:${result.mimeType};base64,${result.data}`}
            alt={`Result from ${output.tool_name}`}
            className="mt-2 max-h-96 max-w-full rounded object-contain"
          />
        ) : isJson ? (
          <ContentCodeBlock className="mt-2 max-h-96 overflow-y-auto">
            {resultText}
          </ContentCodeBlock>
        ) : (
          <p className="mt-2 max-h-96 overflow-y-auto whitespace-pre-wrap break-words text-sm text-zinc-800">
            {resultText}
          </p>
        )}
      </ContentCard>
    </ContentGrid>
  );
}
