import { useMainDashboardPage } from "./useMainDashboardPage";
import { Separator } from "@/components/ui/separator";
import { AgentTable } from "../AgentTable/AgentTable";
import { StatusType } from "@/components/agptui/Status";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { EmptySubmissions } from "./components/EmptySubmissions";
import { SubmissionLoadError } from "./components/SumbmissionLoadError";
import { SubmissionsLoading } from "./components/SubmissionsLoading";

export const MainDashboardPage = () => {
  const {
    onDeleteSubmission,
    onEditSubmission,
    onOpenSubmitModal,
    onPublishStateChange,
    publishState,
    // API data
    submissions,
    isLoading,
    error,
  } = useMainDashboardPage();

  return (
    <main className="flex-1 py-8">
      {/* Header Section */}
      <div className="mb-8 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="space-y-6">
          <h1 className="text-4xl font-medium text-neutral-900 dark:text-neutral-100">
            Agent dashboard
          </h1>
          <div className="space-y-2">
            <h2 className="text-xl font-medium text-neutral-900 dark:text-neutral-100">
              Submit a New Agent
            </h2>
            <p className="text-sm text-[#707070] dark:text-neutral-400">
              Select from the list of agents you currently have, or upload from
              your local machine.
            </p>
          </div>
        </div>
        <PublishAgentModal
          targetState={publishState}
          onStateChange={onPublishStateChange}
          trigger={
            <Button size="small" onClick={onOpenSubmitModal}>
              Submit agent
            </Button>
          }
        />
      </div>

      <Separator className="mb-8" />

      {/* Agents Section */}
      <div>
        <h2 className="mb-4 text-xl font-bold text-neutral-900 dark:text-neutral-100">
          Your uploaded agents
        </h2>

        {error ? (
          <SubmissionLoadError />
        ) : isLoading ? (
          <SubmissionsLoading />
        ) : submissions && submissions.submissions.length > 0 ? (
          <AgentTable
            agents={submissions.submissions.map((submission, index) => ({
              id: index,
              agent_id: submission.agent_id,
              agent_version: submission.agent_version,
              sub_heading: submission.sub_heading,
              date_submitted: submission.date_submitted,
              agentName: submission.name,
              description: submission.description,
              imageSrc: submission.image_urls || [""],
              dateSubmitted: new Date(
                submission.date_submitted,
              ).toLocaleDateString(),
              status: submission.status.toLowerCase() as StatusType,
              runs: submission.runs,
              rating: submission.rating,
            }))}
            onEditSubmission={onEditSubmission}
            onDeleteSubmission={onDeleteSubmission}
          />
        ) : (
          <EmptySubmissions />
        )}
      </div>
    </main>
  );
};
