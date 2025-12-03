import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { formatDate } from "@/lib/utils/time";
import { RunAgentModal } from "../modals/RunAgentModal/RunAgentModal";
import { RunDetailCard } from "../selected-views/RunDetailCard/RunDetailCard";
import { EmptyTasksIllustration } from "./EmptyTasksIllustration";

type Props = {
  agent: LibraryAgent;
};

export function EmptyTasks({ agent }: Props) {
  const isPublished = Boolean(agent.marketplace_listing);
  const createdAt = formatDate(agent.created_at);
  const updatedAt = formatDate(agent.updated_at);
  const isUpdated = updatedAt !== createdAt;

  return (
    <div className="my-4 flex min-h-0 flex-1 flex-col gap-2 px-2 lg:flex-row">
      <RunDetailCard className="relative flex min-h-0 flex-1 flex-col overflow-hidden border-none">
        <div className="flex flex-1 flex-col items-center justify-center gap-0">
          <EmptyTasksIllustration className="-mt-20" />
          <div className="flex flex-col items-center gap-12">
            <div className="flex items-center justify-between gap-2">
              <div className="flex flex-col items-center gap-2">
                <Text variant="h3" className="text-center text-[1.375rem]">
                  Ready to get started?
                </Text>
                <Text variant="large" className="text-center">
                  Run your agent and this space will fill with your agent&apos;s
                  activity
                </Text>
              </div>
            </div>
            <RunAgentModal
              triggerSlot={
                <Button
                  variant="primary"
                  size="large"
                  className="inline-flex w-[19.75rem]"
                >
                  Setup your task
                </Button>
              }
              agent={agent}
              agentId={agent.id.toString()}
            />
          </div>
        </div>
      </RunDetailCard>

      <div className="mt-4 flex flex-col gap-10 rounded-large border border-zinc-200 p-6 lg:mt-0 lg:w-[29.5rem]">
        <Text variant="label" className="text-zinc-500">
          About this agent
        </Text>
        <div className="flex flex-col gap-2">
          <Text variant="h4">{agent.name}</Text>
          {isPublished ? (
            <Text variant="body">
              by {agent.marketplace_listing?.creator.name}
            </Text>
          ) : null}
        </div>
        <ShowMoreText
          previewLimit={170}
          variant="body"
          className="-mt-4 text-textGrey"
        >
          {agent.description ||
            `This agent is not yet published. Once it is published, You can publish your agent by clicking the "Publish" button in the agent editor.`}
        </ShowMoreText>
        <div className="flex flex-col gap-4">
          <div className="flex items-center gap-20">
            <div className="flex flex-col gap-0">
              <Text variant="body-medium" className="text-black">
                Agent created on
              </Text>
              <Text variant="body" className="text-textGrey">
                {createdAt}
              </Text>
            </div>
            {isUpdated ? (
              <div className="flex flex-col gap-0">
                <Text variant="body-medium" className="text-black">
                  Agent updated on
                </Text>
                <Text variant="body" className="text-textGrey">
                  {updatedAt}
                </Text>
              </div>
            ) : null}
          </div>
          <div className="mt-4 flex items-center gap-2">
            <Button variant="secondary" size="small">
              Edit agent
            </Button>
            <Button variant="secondary" size="small">
              Export agent to file
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
