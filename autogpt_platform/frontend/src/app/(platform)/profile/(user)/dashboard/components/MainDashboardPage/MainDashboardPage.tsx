import { useMainDashboardPage } from "./useMainDashboardPage";
import { Separator } from "@/components/ui/separator";
import { AgentTable } from "../AgentTable/AgentTable";
import { PublishAgentModal } from "@/components/contextual/PublishAgentModal/PublishAgentModal";
import { EditAgentModal } from "@/components/contextual/EditAgentModal/EditAgentModal";
import { Button } from "@/components/atoms/Button/Button";
import { EmptySubmissions } from "./components/EmptySubmissions";
import { SubmissionLoadError } from "./components/SumbmissionLoadError";
import { SubmissionsLoading } from "./components/SubmissionsLoading";
import { Text } from "@/components/atoms/Text/Text";

export const MainDashboardPage = () => {
  const {
    onDeleteSubmission,
    onViewSubmission,
    onEditSubmission,
    onEditSuccess,
    onEditClose,
    onOpenSubmitModal,
    onPublishStateChange,
    publishState,
    editState,
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
          <Text variant="h1" size="h3">
            Agent dashboard
          </Text>
          <div className="space-y-2">
            <Text
              variant="h2"
              size="large-medium"
              className="text-neutral-900 dark:text-neutral-100"
            >
              Submit a New Agent
            </Text>
            <Text variant="body" size="small">
              Select from the list of agents you currently have, or upload from
              your local machine.
            </Text>
          </div>
        </div>
        <PublishAgentModal
          targetState={publishState}
          onStateChange={onPublishStateChange}
          trigger={
            <Button
              data-testid="submit-agent-button"
              size="small"
              onClick={onOpenSubmitModal}
            >
              Submit agent
            </Button>
          }
        />
      </div>

      <Separator className="mb-8" />

      {/* Agents Section */}
      <div>
        <Text
          variant="h2"
          size="large-medium"
          className="mb-4 text-neutral-900 dark:text-neutral-100"
        >
          Your uploaded agents
        </Text>

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
              status: submission.status,
              runs: submission.runs,
              rating: submission.rating,
              video_url: submission.video_url || undefined,
              categories: submission.categories,
              slug: submission.slug,
              store_listing_version_id:
                submission.store_listing_version_id || undefined,
            }))}
            onViewSubmission={onViewSubmission}
            onDeleteSubmission={onDeleteSubmission}
            onEditSubmission={onEditSubmission}
          />
        ) : (
          <EmptySubmissions />
        )}
      </div>

      <EditAgentModal
        isOpen={editState.isOpen}
        onClose={onEditClose}
        submission={editState.submission}
        onSuccess={onEditSuccess}
      />
    </main>
  );
};
