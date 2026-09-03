"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { useRemoveWorkflowDialog } from "./useRemoveWorkflowDialog";

type Props = {
  expert: Expert;
  workflow: ExpertWorkflowRef;
  onClose: () => void;
};

export function RemoveWorkflowDialog({ expert, workflow, onClose }: Props) {
  const workflowName = workflow.name ?? "Unnamed workflow";
  const { isRemoving, handleRemove } = useRemoveWorkflowDialog({
    expertId: expert.id,
    expertName: expert.name,
    workflowId: workflow.id,
    workflowName,
    graphID: workflow.graph_id ?? null,
    onClose,
  });

  // Match the disabled Cancel button: while the request is in flight ESC and
  // overlay clicks must not dismiss either, or the user loses the
  // pending/success/failure feedback for a destructive action.
  function handleOpenChange(nextOpen: boolean) {
    if (!nextOpen && !isRemoving) onClose();
  }

  return (
    <Dialog
      controlled={{ isOpen: true, set: handleOpenChange }}
      styling={{ maxWidth: "28rem" }}
      title={`Remove ${workflowName}?`}
    >
      <Dialog.Content>
        <div className="flex flex-col gap-4">
          <Text variant="body" className="text-zinc-600">
            {expert.name} stops running it and any schedule for it is cancelled.
            The agent stays in your library, so you can add it back anytime.
          </Text>
          <Dialog.Footer>
            <Button variant="secondary" disabled={isRemoving} onClick={onClose}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              loading={isRemoving}
              onClick={handleRemove}
              data-testid="remove-workflow-confirm"
            >
              Remove
            </Button>
          </Dialog.Footer>
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
