import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  ArrowBendLeftUpIcon,
  ArrowBendRightDownIcon,
  CardsThreeIcon,
  EyeIcon,
  StopIcon,
} from "@phosphor-icons/react";
import { AgentActionsDropdown } from "../../../AgentActionsDropdown";
import { SelectedActionsWrap } from "../../../SelectedActionsWrap";
import { ShareRunButton } from "../../../ShareRunButton/ShareRunButton";
import { CreateTemplateModal } from "../CreateTemplateModal/CreateTemplateModal";
import { useSelectedRunActions } from "./useSelectedRunActions";

type Props = {
  agent: LibraryAgent;
  run: GraphExecution | undefined;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
};

export function SelectedRunActions({
  agent,
  run,
  onSelectRun,
  onClearSelectedRun,
}: Props) {
  const {
    canRunManually,
    handleRunAgain,
    handleStopRun,
    isRunningAgain,
    canStop,
    isStopping,
    openInBuilderHref,
    handleCreateTemplate,
    isCreateTemplateModalOpen,
    setIsCreateTemplateModalOpen,
  } = useSelectedRunActions({
    agentGraphId: agent.graph_id,
    run: run,
    agent: agent,
    onSelectRun: onSelectRun,
  });

  const shareExecutionResultsEnabled = useGetFlag(Flag.SHARE_EXECUTION_RESULTS);
  const isRunning = run?.status === "RUNNING";

  if (!run || !agent) return null;

  return (
    <SelectedActionsWrap>
      {canRunManually && !isRunning ? (
        <Button
          variant="icon"
          size="icon"
          aria-label="Rerun task"
          onClick={handleRunAgain}
          disabled={isRunningAgain}
        >
          {isRunningAgain ? (
            <LoadingSpinner size="small" />
          ) : (
            <div className="gap- relative flex flex-col items-center justify-center">
              <ArrowBendLeftUpIcon
                weight="bold"
                size={16}
                className="relative bottom-[4px] z-0 rotate-90 text-zinc-700"
              />
              <ArrowBendRightDownIcon
                weight="bold"
                size={16}
                className="absolute bottom-[-5px] z-10 rotate-90 text-zinc-700"
              />
            </div>
          )}
        </Button>
      ) : null}
      {canStop ? (
        <Button
          variant="icon"
          size="icon"
          aria-label="Stop task"
          onClick={handleStopRun}
          disabled={isStopping}
          className="border-red-600 bg-red-600 text-white hover:border-red-800 hover:bg-red-800"
        >
          <StopIcon weight="bold" size={18} />
        </Button>
      ) : null}
      {openInBuilderHref ? (
        <Button
          variant="icon"
          size="icon"
          as="NextLink"
          href={openInBuilderHref}
          target="_blank"
          aria-label="View task details"
        >
          <EyeIcon weight="bold" size={18} className="text-zinc-700" />
        </Button>
      ) : null}
      {shareExecutionResultsEnabled && (
        <ShareRunButton
          graphId={agent.graph_id}
          executionId={run.id}
          isShared={run.is_shared}
          shareToken={run.share_token}
        />
      )}
      {canRunManually && (
        <>
          <Button
            variant="icon"
            size="icon"
            aria-label="Save task as template"
            onClick={() => setIsCreateTemplateModalOpen(true)}
            title="Create template"
          >
            <CardsThreeIcon weight="bold" size={18} className="text-zinc-700" />
          </Button>
          <CreateTemplateModal
            isOpen={isCreateTemplateModalOpen}
            onClose={() => setIsCreateTemplateModalOpen(false)}
            onCreate={handleCreateTemplate}
            run={run}
          />
        </>
      )}
      <AgentActionsDropdown
        agent={agent}
        run={run}
        agentGraphId={agent.graph_id}
        onClearSelectedRun={onClearSelectedRun}
      />
    </SelectedActionsWrap>
  );
}
