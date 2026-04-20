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
import { useCopilotChatActions } from "../../../../components/CopilotChatActionsProvider/useCopilotChatActions";

interface Props {
  output: ExecutionStartedResponse;
}

export function ExecutionStartedCard({ output }: Props) {
  const router = useRouter();
  // In the builder panel the run_agent effect already drops the exec_id
  // onto the URL so the builder's in-place execution UI opens — the
  // "View Execution" button here would navigate the user away from the
  // page they're editing, so hide it.
  const { chatSurface } = useCopilotChatActions();
  const hideViewExecution = chatSurface === "builder";

  return (
    <ContentGrid>
      <ContentCard>
        <ContentCardTitle>Execution started</ContentCardTitle>
        <ContentCardSubtitle>{output.execution_id}</ContentCardSubtitle>
        <ContentCardDescription>{output.message}</ContentCardDescription>
        {!hideViewExecution && (
          <Button
            size="small"
            className="mt-3"
            onClick={() =>
              router.push(
                output.library_agent_link ??
                  `/library/agents/${output.graph_id}?activeTab=runs&activeItem=${output.execution_id}`,
              )
            }
          >
            View Execution
          </Button>
        )}
      </ContentCard>
    </ContentGrid>
  );
}
