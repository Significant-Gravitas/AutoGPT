"use client";

import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { AgentInfoStep } from "../PublishAgentModal/components/AgentInfoStep/AgentInfoStep";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { useGetV2GetASingleSubmission } from "@/app/api/__generated__/endpoints/store/store";

export interface EditAgentModalProps {
  isOpen: boolean;
  onClose: () => void;
  submission: StoreSubmission | null;
  onSuccess: () => void;
}

export function EditAgentModal({
  isOpen,
  onClose,
  submission,
  onSuccess,
}: EditAgentModalProps) {
  const { data: detailedSubmission, isLoading, error } = useGetV2GetASingleSubmission(
    submission?.store_listing_version_id || "",
    {
      query: {
        enabled: isOpen && !!submission?.store_listing_version_id,
      },
    }
  );

  if (!submission) return null;

  const submissionData = (detailedSubmission?.status === 200 ? detailedSubmission.data : submission) as StoreSubmission;

  const initialData = {
    agent_id: submissionData.agent_id,
    title: submissionData.name,
    subheader: submissionData.sub_heading,
    slug: submissionData.slug,
    thumbnailSrc: submissionData.image_urls?.[0] || "",
    youtubeLink: submissionData.video_url || "",
    category: submissionData.categories?.join(", ") || "",
    description: submissionData.description,
    additionalImages: submissionData.image_urls?.slice(1) || [],
  };

  return (
    <Dialog
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
          {isLoading ? (
            <div className="flex min-h-[60vh] flex-col items-center justify-center gap-8 space-y-2">
              <div className="text-lg">Loading submission details...</div>
            </div>
          ) : error ? (
            <div className="flex min-h-[60vh] flex-col items-center justify-center gap-8 space-y-2">
              <div className="text-lg text-red-600">Error loading submission details</div>
            </div>
          ) : (
            <AgentInfoStep
              onBack={onClose}
              onSuccess={onSuccess}
              selectedAgentId={submissionData.agent_id}
              selectedAgentVersion={submissionData.agent_version}
              initialData={initialData}
              isEditing={true}
              store_listing_version_id={submissionData.store_listing_version_id || undefined}
            />
          )}
        </div>
      </Dialog.Content>
    </Dialog>
  );
} 