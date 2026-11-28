import { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { RunAgentModal } from "../../../../modals/RunAgentModal/RunAgentModal";
import { AgentActionsDropdown } from "../../../AgentActionsDropdown";
import { SelectedActionsWrap } from "../../../SelectedActionsWrap";
import { ShareRunButton } from "../../../ShareRunButton/ShareRunButton";
import { CreateTemplateModal } from "../CreateTemplateModal/CreateTemplateModal";
import { DeletedKeysNotice } from "./components/DeletedKeysNotice";
import { RunVersionErrorDialog } from "./components/RunVersionErrorDialog";
import { RERUN_FORM_TITLE } from "./helpers";
import { useSelectedRunActions } from "./useSelectedRunActions";
import {
  Album01Icon,
  CornerLeftUpIcon,
  CornerRightDownIcon,
  EyeIcon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  agent: LibraryAgent;
  run: GraphExecution | undefined;
  onSelectRun?: (id: string) => void;
  onScheduleCreated?: (schedule: GraphExecutionJobInfo) => void;
  onClearSelectedRun?: () => void;
};

export function SelectedRunActions({
  agent,
  run,
  onSelectRun,
  onScheduleCreated,
  onClearSelectedRun,
}: Props) {
  const {
    canRunManually,
    handleRunAgain,
    rerunForm,
    rerunAgent,
    isOpeningRerunForm,
    hasRerunVersionError,
    isRetryingRerunVersion,
    retryRerunVersion,
    handleRerunFormOpenChange,
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

  const isRunning = run?.status === "RUNNING";
  const isRerunBusy = isRunningAgain || isOpeningRerunForm;

  if (!run || !agent) return null;

  return (
    <SelectedActionsWrap>
      {canRunManually && !isRunning ? (
        <Button
          variant="icon"
          size="icon"
          aria-label="Rerun task"
          onClick={handleRunAgain}
          disabled={isRerunBusy}
        >
          {isRerunBusy ? (
            <LoadingSpinner size="small" />
          ) : (
            <div className="relative flex flex-col items-center justify-center">
              <Icon
                icon={CornerLeftUpIcon}
                size={16}
                className="relative bottom-[4px] z-base rotate-90"
              />
              <Icon
                icon={CornerRightDownIcon}
                size={16}
                className="absolute bottom-[-5px] z-raised rotate-90"
              />
            </div>
          )}
        </Button>
      ) : null}
      {rerunForm && rerunAgent ? (
        <RunAgentModal
          agent={rerunAgent}
          title={RERUN_FORM_TITLE}
          controlled={{ isOpen: true, set: handleRerunFormOpenChange }}
          notice={<DeletedKeysNotice deletedKeys={rerunForm.deletedKeys} />}
          initialInputValues={rerunForm.inputs}
          initialInputCredentials={rerunForm.credentials}
          fieldsToPick={rerunForm.fieldsToPick}
          keepsSkippedCredentials
          onRunCreated={(execution) => onSelectRun?.(execution.id)}
          onScheduleCreated={onScheduleCreated}
        />
      ) : null}
      {rerunForm && hasRerunVersionError ? (
        <RunVersionErrorDialog
          version={run.graph_version}
          deletedKeys={rerunForm.deletedKeys}
          isRetrying={isRetryingRerunVersion}
          onRetry={retryRerunVersion}
          onOpenChange={handleRerunFormOpenChange}
        />
      ) : null}
      {canStop ? (
        <Button
          variant="destructive"
          size="icon"
          aria-label="Stop task"
          onClick={handleStopRun}
          disabled={isStopping}
        >
          <Icon icon={StopIcon} size={18} />
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
          <Icon icon={EyeIcon} size={18} />
        </Button>
      ) : null}
      <ShareRunButton
        graphId={agent.graph_id}
        executionId={run.id}
        isShared={run.is_shared}
        shareToken={run.share_token}
      />
      {canRunManually && (
        <>
          <Button
            variant="icon"
            size="icon"
            aria-label="Save task as template"
            onClick={() => setIsCreateTemplateModalOpen(true)}
            title="Create template"
          >
            <Icon icon={Album01Icon} size={18} />
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
