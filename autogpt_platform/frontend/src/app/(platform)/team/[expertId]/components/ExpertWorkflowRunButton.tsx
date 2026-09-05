"use client";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/RunAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";

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
      onRunCreated={onRunCreated}
      onTriggerSetup={onTriggerSetup}
      triggerSlot={
        <Button
          type="button"
          variant={variant}
          size="small"
          className={ACTION_BUTTON_CLASS}
          leftIcon={<Icon icon={PlayIcon} size={14} />}
        >
          {isTriggerWorkflow ? "Set up trigger" : "Run"}
        </Button>
      }
    />
  );
}
