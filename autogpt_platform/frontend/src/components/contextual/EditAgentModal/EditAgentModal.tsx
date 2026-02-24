"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { EditAgentForm } from "./components/EditAgentForm";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";

export interface EditAgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  submission:
    | (StoreSubmissionEditRequest & {
        store_listing_version_id: string | undefined;
        agent_id: string;
      })
    | null;
  onSuccess: (submission: StoreSubmission) => void;
}

export function EditAgentModal({
  isOpen,
  onClose,
  submission,
  onSuccess,
}: EditAgentModalProps) {
  if (!submission) return null;

  return (
    <Dialog
      title="Edit Agent"
      styling={{
        maxWidth: "45rem",
      }}
      controlled={{
        isOpen,
        set: (isOpen) => {
          if (!isOpen) onClose();
        },
      }}
    >
      <Dialog.Content>
        <div data-testid="edit-agent-modal">
          <EditAgentForm
            submission={submission}
            onClose={onClose}
            onSuccess={onSuccess}
          />
        </div>
      </Dialog.Content>
    </Dialog>
  );
}
