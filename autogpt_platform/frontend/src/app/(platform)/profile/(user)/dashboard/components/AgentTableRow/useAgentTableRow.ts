import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

interface useAgentTableRowProps {
  id: number;
  onViewSubmission: (submission: StoreSubmission) => void;
  onDeleteSubmission: (submission_id: string) => void;
  agent_id: string;
  agent_version: number;
  agentName: string;
  sub_heading: string;
  description: string;
  imageSrc: string[];
  dateSubmitted: string;
  status: string;
  runs: number;
  rating: number;
}

export const useAgentTableRow = ({
  onViewSubmission,
  onDeleteSubmission,
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
      // SafeCast: status is a string from the API...
      status: status.toUpperCase() as SubmissionStatus,
      runs,
      rating,
    } satisfies StoreSubmission);
  };

  const handleDelete = () => {
    onDeleteSubmission(agent_id);
  };

  return { handleView, handleDelete };
};
