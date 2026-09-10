"use client";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/RunAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { PlayIcon } from "@hugeicons/core-free-icons";

interface Props {
  agent: LibraryAgent;
  isTriggerWorkflow: boolean;
  variant?: "primary" | "secondary";
  onRunCreated: (execution: GraphExecutionMeta) => void;
  onTriggerSetup: () => void;
}

export function ExpertWorkflowRunButton({
  agent,
  isTriggerWorkflow,
  onRunCreated,
  onTriggerSetup,
  variant = "secondary",
}: Props) {
  return (
    <RunAgentModal
      agent={agent}
      title={isTriggerWorkflow ? "Set up trigger" : "Run workflow"}
      dialogVariant="compact"
      onRunCreated={onRunCreated}
      onTriggerSetup={onTriggerSetup}
      triggerSlot={
        <Button
          type="button"
          variant={variant}
          size="xs"
          leadingIcon={PlayIcon}
        >
          {isTriggerWorkflow ? "Set up trigger" : "Run"}
        </Button>
      }
    />
  );
}
