"use client";

import type { ErrorResponse } from "@/app/api/__generated__/models/errorResponse";
import {
  ContentCodeBlock,
  ContentGrid,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import { formatMaybeJson } from "../../helpers";

interface Props {
  output: ErrorResponse;
}

export function ErrorCard({ output }: Props) {
  return (
    <ContentGrid>
      <ContentMessage>{output.message}</ContentMessage>
      {output.error && (
        <ContentCodeBlock>{formatMaybeJson(output.error)}</ContentCodeBlock>
      )}
      {output.details && (
        <ContentCodeBlock>{formatMaybeJson(output.details)}</ContentCodeBlock>
      )}
    </ContentGrid>
  );
}
