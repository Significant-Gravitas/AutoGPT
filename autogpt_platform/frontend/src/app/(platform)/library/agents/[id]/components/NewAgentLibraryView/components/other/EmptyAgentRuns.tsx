import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { formatDate } from "@/lib/utils/time";
import { RunAgentModal } from "../modals/RunAgentModal/RunAgentModal";
import { RunDetailCard } from "../selected-views/RunDetailCard/RunDetailCard";
import { EmptyRunsIllustration } from "./EmptyRunsIllustration";

type Props = {
  agent: LibraryAgent;
};

export function EmptyAgentRuns({ agent }: Props) {
  const isPublished = Boolean(agent.marketplace_listing);
  const createdAt = formatDate(agent.created_at);
  const updatedAt = formatDate(agent.updated_at);
  const isUpdated = updatedAt !== createdAt;

  return (
    <div className="min-h-0 flex-1 flex-col flex-nowrap gap-2 px-2 lg:flex lg:flex-row">
      <RunDetailCard className="relative flex min-h-0 flex-1 flex-col overflow-hidden">
        <div className="flex flex-1 flex-col items-center justify-center gap-0">
          <EmptyRunsIllustration className="-mt-20" />
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
      {isPublished ? (
        <div className="mt-4 flex flex-col gap-10 rounded-large border border-zinc-200 p-6 lg:mt-0 lg:w-[29.5rem]">
          <Text variant="label" className="text-zinc-500">
            About this agent
          </Text>
          <div className="flex flex-col gap-2">
            <Text variant="h4">{agent.name}</Text>
            <Text variant="body">
              by{" "}
              <Link
                href={`/marketplace/creator/${agent.marketplace_listing?.creator.slug}`}
                variant="secondary"
              >
                {agent.marketplace_listing?.creator.name}
              </Link>
            </Text>
          </div>
          <ShowMoreText previewLimit={170} variant="body" className="-mt-4">
            {agent.description ||
              `Note: If you're using Docker Compose watch mode (docker compose watch), it will automatically rebuild on file changes. Since you're using docker compose up -d, manual rebuilds are needed.
You can test the endpoint from your frontend; it should return the marketplace_listing field when an agent has been published, or null if it hasn't.`}
          </ShowMoreText>
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-20">
              <div className="flex flex-col gap-2">
                <Text variant="body-medium" className="text-black">
                  Agent created on
                </Text>
                <Text variant="body">{createdAt}</Text>
              </div>
              {isUpdated ? (
                <div className="flex flex-col gap-2">
                  <Text variant="body-medium" className="text-black">
                    Agent updated on
                  </Text>
                  <Text variant="body">{updatedAt}</Text>
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
      ) : null}
    </div>
  );
}
