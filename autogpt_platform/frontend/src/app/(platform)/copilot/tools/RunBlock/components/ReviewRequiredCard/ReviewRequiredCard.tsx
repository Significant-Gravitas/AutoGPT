"use client";

import {
  ContentCodeBlock,
  ContentGrid,
  ContentMessage,
} from "../../../../components/ToolAccordion/AccordionContent";
import { formatMaybeJson } from "../../helpers";
import type { ReviewRequiredResponse } from "../../helpers";

interface Props {
  output: ReviewRequiredResponse;
}

export function ReviewRequiredCard({ output }: Props) {
  return (
    <ContentGrid>
      <ContentMessage>{output.message}</ContentMessage>
      {Object.keys(output.input_data).length > 0 && (
        <ContentCodeBlock>
          {formatMaybeJson(output.input_data)}
        </ContentCodeBlock>
      )}
    </ContentGrid>
  );
}
