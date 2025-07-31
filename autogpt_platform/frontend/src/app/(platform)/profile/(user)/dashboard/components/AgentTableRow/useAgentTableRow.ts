import { StoreSubmissionRequest } from "@/app/api/__generated__/models/storeSubmissionRequest";

interface useAgentTableRowProps {
  id: number;
  onEditSubmission: (submission: StoreSubmissionRequest) => void;
  onDeleteSubmission: (submission_id: string) => void;
  agent_id: string;
  agent_version: number;
  agentName: string;
  sub_heading: string;
  description: string;
  imageSrc: string[];
}

export const useAgentTableRow = ({
  onEditSubmission,
  onDeleteSubmission,
  agent_id,
  agent_version,
  agentName,
  sub_heading,
  description,
  imageSrc,
}: useAgentTableRowProps) => {
  const handleEdit = () => {
    onEditSubmission({
      agent_id,
      agent_version,
      slug: "",
      name: agentName,
      sub_heading,
      description,
      image_urls: imageSrc,
      categories: [],
    } satisfies StoreSubmissionRequest);
  };

  const handleDelete = () => {
    onDeleteSubmission(agent_id);
  };

  return { handleEdit, handleDelete };
};
