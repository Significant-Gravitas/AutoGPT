import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

interface useAgentTableRowProps {
  id: number;
  onViewSubmission: (submission: StoreSubmission) => void;
  onDeleteSubmission: (submission_id: string) => void;
  onEditSubmission: (
    submission: StoreSubmissionEditRequest & {
      store_listing_version_id: string | undefined;
      agent_id: string;
    },
  ) => void;
  agent_id: string;
  agent_version: number;
  agentName: string;
  sub_heading: string;
  description: string;
  imageSrc: string[];
  dateSubmitted: Date;
  status: SubmissionStatus;
  runs: number;
  rating: number;
  video_url?: string;
  categories?: string[];
  store_listing_version_id?: string;
}

export const useAgentTableRow = ({
  onViewSubmission,
  onDeleteSubmission,
  onEditSubmission,
  agent_id,
  agent_version,
  agentName,
  sub_heading,
  description,
  imageSrc,
  dateSubmitted,
  status,
  runs,
  rating,
  video_url,
  categories,
  store_listing_version_id,
}: useAgentTableRowProps) => {
  const handleView = () => {
    onViewSubmission({
      agent_id,
      agent_version,
      slug: "",
      name: agentName,
      sub_heading,
      description,
      image_urls: imageSrc,
      date_submitted: dateSubmitted,
      status: status,
      runs,
      rating,
      video_url,
      categories,
      store_listing_version_id,
    } satisfies StoreSubmission);
  };

  const handleEdit = () => {
    onEditSubmission({
      name: agentName,
      sub_heading,
      description,
      image_urls: imageSrc,
      video_url,
      categories,
      changes_summary: "Update Submission",
      store_listing_version_id,
      agent_id,
    });
  };

  const handleDelete = () => {
    onDeleteSubmission(agent_id);
  };

  return { handleView, handleDelete, handleEdit };
};
