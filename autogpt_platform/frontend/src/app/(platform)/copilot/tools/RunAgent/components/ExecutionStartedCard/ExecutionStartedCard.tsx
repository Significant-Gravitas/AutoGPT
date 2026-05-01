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

export function titleForStatus(status: string | undefined): string {
  // Normalise whatever the backend sent (QUEUED/RUNNING/COMPLETED/FAILED/
  // STOPPED/TERMINATED/TIMED_OUT/INCOMPLETE/CANCELLED …). The card is
  // reused for both truly-just-queued runs and for sync-completed runs
  // (run_agent with wait_for_result) — "Execution started" is wrong for
  // the latter.
  const s = (status ?? "").toUpperCase();
  if (s === "COMPLETED") return "Execution completed";
  if (s === "FAILED") return "Execution failed";
  if (s === "STOPPED" || s === "TERMINATED" || s === "CANCELLED")
    return "Execution stopped";
  if (s === "TIMED_OUT" || s === "INCOMPLETE") return "Execution incomplete";
  if (s === "RUNNING") return "Execution running";
  return "Execution started";
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
        <ContentCardTitle>{titleForStatus(output.status)}</ContentCardTitle>
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
