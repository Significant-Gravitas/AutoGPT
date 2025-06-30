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
  selectedAgents: Set<string>;
  setSelectedAgents: React.Dispatch<React.SetStateAction<Set<string>>>;
}

export const useAgentTableRow = ({
  id,
  onEditSubmission,
  onDeleteSubmission,
  agent_id,
  agent_version,
  agentName,
  sub_heading,
  description,
  imageSrc,
  selectedAgents,
  setSelectedAgents,
}: useAgentTableRowProps) => {
  const checkboxId = `agent-${id}-checkbox`;

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

  const handleCheckboxChange = () => {
    if (selectedAgents.has(agent_id)) {
      selectedAgents.delete(agent_id);
    } else {
      selectedAgents.add(agent_id);
    }
    setSelectedAgents(new Set(selectedAgents));
  };

  return { checkboxId, handleEdit, handleDelete, handleCheckboxChange };
};
