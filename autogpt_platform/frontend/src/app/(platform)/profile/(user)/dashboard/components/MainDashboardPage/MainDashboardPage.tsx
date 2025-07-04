import { PublishAgentPopout } from "@/components/agptui/composite/PublishAgentPopout";
import { Button } from "@/components/ui/button";
import { useMainDashboardPage } from "./useMainDashboardPage";
import { Separator } from "@/components/ui/separator";
import { AgentTable } from "../AgentTable/AgentTable";
import { StatusType } from "@/components/agptui/Status";

export const MainDashboardPage = () => {
  const {
    onOpenPopout,
    onDeleteSubmission,
    onEditSubmission,
    submissions,
    isLoading,
    openPopout,
    submissionData,
    popoutStep,
  } = useMainDashboardPage();

  if (isLoading) {
    return "Loading....";
  }

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
        <PublishAgentPopout
          trigger={
            <Button
              onClick={onOpenPopout}
              className="h-9 rounded-full bg-black px-4 text-sm font-medium text-white hover:bg-neutral-700 dark:hover:bg-neutral-600"
            >
              Submit agent
            </Button>
          }
          openPopout={openPopout}
          inputStep={popoutStep}
          submissionData={submissionData}
        />
      </div>

      <Separator className="mb-8" />

      {/* Agents Section */}
      <div>
        <h2 className="mb-4 text-xl font-bold text-neutral-900 dark:text-neutral-100">
          Your uploaded agents
        </h2>
        {submissions && (
          <AgentTable
            agents={
              submissions?.submissions.map((submission, index) => ({
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
              })) || []
            }
            onEditSubmission={onEditSubmission}
            onDeleteSubmission={onDeleteSubmission}
          />
        )}
      </div>
    </main>
  );
};
