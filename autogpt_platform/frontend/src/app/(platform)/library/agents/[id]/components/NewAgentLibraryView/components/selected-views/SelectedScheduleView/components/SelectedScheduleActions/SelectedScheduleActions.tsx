"use client";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { EyeIcon, Play, TrashIcon } from "@phosphor-icons/react";
import { AgentActionsDropdown } from "../../../AgentActionsDropdown";
import { SelectedActionsWrap } from "../../../SelectedActionsWrap";
import { useSelectedScheduleActions } from "./useSelectedScheduleActions";

type Props = {
  agent: LibraryAgent;
  scheduleId: string;
  schedule?: GraphExecutionJobInfo;
  onDeleted?: () => void;
  onSelectRun?: (id: string) => void;
};

export function SelectedScheduleActions({
  agent,
  scheduleId,
  schedule,
  onDeleted,
  onSelectRun,
}: Props) {
  const {
    openInBuilderHref,
    showDeleteDialog,
    setShowDeleteDialog,
    handleDelete,
    isDeleting,
    handleRunNow,
    isRunning,
  } = useSelectedScheduleActions({
    agent,
    scheduleId,
    schedule,
    onDeleted,
    onSelectRun,
  });

  return (
    <>
      <SelectedActionsWrap>
        <Button
          variant="icon"
          size="icon"
          aria-label="Run now"
          onClick={handleRunNow}
          disabled={isRunning || !schedule}
        >
          {isRunning ? (
            <LoadingSpinner size="small" />
          ) : (
            <Play weight="bold" size={18} className="text-zinc-700" />
          )}
        </Button>
        {openInBuilderHref && (
          <Button
            variant="icon"
            size="icon"
            as="NextLink"
            href={openInBuilderHref}
            target="_blank"
            aria-label="View scheduled task details"
          >
            <EyeIcon weight="bold" size={18} className="text-zinc-700" />
          </Button>
        )}
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete schedule"
          onClick={() => setShowDeleteDialog(true)}
          disabled={isDeleting}
        >
          {isDeleting ? (
            <LoadingSpinner size="small" />
          ) : (
            <TrashIcon weight="bold" size={18} />
          )}
        </Button>
        <AgentActionsDropdown agent={agent} scheduleId={scheduleId} />
      </SelectedActionsWrap>

      <Dialog
        controlled={{
          isOpen: showDeleteDialog,
          set: setShowDeleteDialog,
        }}
        styling={{ maxWidth: "32rem" }}
        title="Delete schedule"
      >
        <Dialog.Content>
          <Text variant="large">
            Are you sure you want to delete this schedule? This action cannot be
            undone.
          </Text>
          <Dialog.Footer>
            <Button
              variant="secondary"
              onClick={() => setShowDeleteDialog(false)}
              disabled={isDeleting}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              loading={isDeleting}
            >
              Delete Schedule
            </Button>
          </Dialog.Footer>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
