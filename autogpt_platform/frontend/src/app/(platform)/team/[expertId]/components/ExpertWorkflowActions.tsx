"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { RunAgentModal } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/modals/RunAgentModal/RunAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  PencilEdit02Icon,
  PlayIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import { ExpertWorkflowCardMenu } from "./ExpertWorkflowCardMenu";
import { ACTION_BUTTON_CLASS } from "@/app/(platform)/team/helpers";

const ICON_BUTTON_CLASS =
  "h-7 w-7 rounded-md border-transparent bg-white/90 p-0 text-zinc-700 hover:border-transparent hover:bg-white";

interface ActionsProps {
  workflow: ExpertWorkflowRef;
  expertId: string;
  name: string;
  builderHref: string | null;
  chatHref: string;
  buttonClassName?: string;
  menuClassName?: string;
}

export function ExpertWorkflowActions({
  workflow,
  expertId,
  name,
  builderHref,
  chatHref,
  buttonClassName = ICON_BUTTON_CLASS,
  menuClassName = buttonClassName,
}: ActionsProps) {
  return (
    <>
      {builderHref ? (
        <Button
          as="NextLink"
          href={builderHref}
          variant="icon"
          size="icon"
          aria-label="Edit workflow"
          className={buttonClassName}
        >
          <Icon icon={PencilEdit02Icon} size={14} />
        </Button>
      ) : null}
      <Button
        as="NextLink"
        href={chatHref}
        variant="icon"
        size="icon"
        aria-label="Ask about this workflow"
        className={buttonClassName}
      >
        <Icon icon={SparklesIcon} size={14} />
      </Button>
      <ExpertWorkflowCardMenu
        workflow={workflow}
        expertId={expertId}
        name={name}
        triggerClassName={menuClassName}
      />
    </>
  );
}

interface RunProps {
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
}: RunProps) {
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
