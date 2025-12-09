import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { FloatingSafeModeToggle } from "@/components/molecules/FloatingSafeModeToggle/FloatingSafeModeToggle";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import {
  ArrowBendLeftUpIcon,
  ArrowBendRightDownIcon,
  CardsThreeIcon,
  EyeIcon,
  StopIcon,
} from "@phosphor-icons/react";
import { AgentActionsDropdown } from "../../../AgentActionsDropdown";
import { ShareRunButton } from "../../../ShareRunButton/ShareRunButton";
import { CreateTemplateModal } from "../CreateTemplateModal/CreateTemplateModal";
import { useSelectedRunActions } from "./useSelectedRunActions";

type Props = {
  agent: LibraryAgent;
  run: GraphExecution | undefined;
  scheduleRecurrence?: string;
  onSelectRun?: (id: string) => void;
  onClearSelectedRun?: () => void;
};

export function SelectedRunActions(props: Props) {
  const {
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
    agentGraphId: props.agent.graph_id,
    run: props.run,
    agent: props.agent,
    onSelectRun: props.onSelectRun,
    onClearSelectedRun: props.onClearSelectedRun,
  });

  const shareExecutionResultsEnabled = useGetFlag(Flag.SHARE_EXECUTION_RESULTS);
  const isRunning = props.run?.status === "RUNNING";

  if (!props.run || !props.agent) return null;

  return (
    <div className="my-4 flex flex-col items-center gap-3">
      {!isRunning ? (
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
          graphId={props.agent.graph_id}
          executionId={props.run.id}
          isShared={props.run.is_shared}
          shareToken={props.run.share_token}
        />
      )}
      <FloatingSafeModeToggle
        graph={props.agent}
        variant="white"
        fullWidth={false}
      />
      <Button
        variant="icon"
        size="icon"
        aria-label="Save task as template"
        onClick={() => setIsCreateTemplateModalOpen(true)}
        title="Create template"
      >
        <CardsThreeIcon weight="bold" size={18} className="text-zinc-700" />
      </Button>
      <AgentActionsDropdown
        agent={props.agent}
        run={props.run}
        agentGraphId={props.agent.graph_id}
        onClearSelectedRun={props.onClearSelectedRun}
      />
      <CreateTemplateModal
        isOpen={isCreateTemplateModalOpen}
        onClose={() => setIsCreateTemplateModalOpen(false)}
        onCreate={handleCreateTemplate}
        run={props.run}
      />
    </div>
  );
}
