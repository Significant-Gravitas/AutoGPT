"use client";

import type { ExecutionStartedResponse } from "@/app/api/__generated__/models/executionStartedResponse";
import { Button } from "@/components/atoms/Button/Button";
import { useRouter } from "next/navigation";
import {
  ContentCard,
  ContentCardDescription,
  ContentCardSubtitle,
  ContentCardTitle,
  ContentGrid,
} from "../../../../components/ToolAccordion/AccordionContent";

interface Props {
  output: ExecutionStartedResponse;
}

export function ExecutionStartedCard({ output }: Props) {
  const router = useRouter();

  return (
    <ContentGrid>
      <ContentCard>
        <ContentCardTitle>Execution started</ContentCardTitle>
        <ContentCardSubtitle>{output.execution_id}</ContentCardSubtitle>
        <ContentCardDescription>{output.message}</ContentCardDescription>
        {output.library_agent_link && (
          <Button
            size="small"
            className="mt-3"
            onClick={() => router.push(output.library_agent_link!)}
          >
            View Execution
          </Button>
        )}
      </ContentCard>
    </ContentGrid>
  );
}
