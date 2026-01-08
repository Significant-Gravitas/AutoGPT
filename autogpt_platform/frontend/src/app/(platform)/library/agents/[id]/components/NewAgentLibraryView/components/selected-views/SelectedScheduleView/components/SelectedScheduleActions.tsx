"use client";

import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { EyeIcon, TrashIcon } from "@phosphor-icons/react";
import { useState } from "react";
import { AgentActionsDropdown } from "../../AgentActionsDropdown";
import { useScheduleDetailHeader } from "../../RunDetailHeader/useScheduleDetailHeader";
import { SelectedActionsWrap } from "../../SelectedActionsWrap";

type Props = {
  agent: LibraryAgent;
  scheduleId: string;
  onDeleted?: () => void;
};

export function SelectedScheduleActions({
  agent,
  scheduleId,
  onDeleted,
}: Props) {
  const { openInBuilderHref, deleteSchedule, isDeleting } =
    useScheduleDetailHeader(agent.graph_id, scheduleId, agent.graph_version);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);

  function handleDelete() {
    deleteSchedule();
    setShowDeleteDialog(false);
    onDeleted?.();
  }

  return (
    <>
      <SelectedActionsWrap>
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
