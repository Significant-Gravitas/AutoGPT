import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";

interface useAgentTableRowProps {
  storeAgentSubmission: StoreSubmission;
  onViewSubmission: (submission: StoreSubmission) => void;
  onDeleteSubmission: (submission_id: string) => void;
  onEditSubmission: (
    submission: StoreSubmissionEditRequest & {
      store_listing_version_id: string | undefined;
      graph_id: string;
    },
  ) => void;
}

export const useAgentTableRow = ({
  storeAgentSubmission,
  onViewSubmission,
  onDeleteSubmission,
  onEditSubmission,
}: useAgentTableRowProps) => {
  const handleView = () => {
    onViewSubmission(storeAgentSubmission);
  };

  const handleEdit = () => {
    onEditSubmission({
      name: storeAgentSubmission.name,
      sub_heading: storeAgentSubmission.sub_heading,
      description: storeAgentSubmission.description,
      image_urls: storeAgentSubmission.image_urls,
      video_url: storeAgentSubmission.video_url,
      categories: storeAgentSubmission.categories,
      changes_summary:
        storeAgentSubmission.changes_summary || "Update Submission",
      store_listing_version_id: storeAgentSubmission.listing_version_id,
      graph_id: storeAgentSubmission.graph_id,
    });
  };

  const handleDelete = () => {
    onDeleteSubmission(storeAgentSubmission.listing_version_id);
  };

  return { handleView, handleDelete, handleEdit };
};
