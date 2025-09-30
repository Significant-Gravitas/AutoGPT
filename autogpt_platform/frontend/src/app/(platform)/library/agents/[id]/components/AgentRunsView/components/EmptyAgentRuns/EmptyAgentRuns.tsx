import { ShowMoreText } from "@/components/molecules/ShowMoreText/ShowMoreText";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { Text } from "@/components/atoms/Text/Text";
import { RunAgentModal } from "../RunAgentModal/RunAgentModal";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { PlusIcon } from "@phosphor-icons/react";

type Props = {
  agentName: string;
  creatorName: string;
  description: string;
  agent: LibraryAgent;
};

export function EmptyAgentRuns({
  agentName,
  creatorName,
  description,
  agent,
}: Props) {
  const isUnknownCreator = creatorName === "Unknown";

  return (
    <div className="mt-6 px-2">
      <RunDetailCard className="relative min-h-[70vh]">
        <div className="absolute left-1/2 top-1/2 flex w-[80%] -translate-x-1/2 -translate-y-1/2 flex-col gap-6 md:w-[60%] lg:w-auto">
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <Text
                variant="h3"
                className="truncate text-ellipsis !font-normal"
              >
                {agentName}
              </Text>
              {!isUnknownCreator ? (
                <Text variant="body-medium">by {creatorName}</Text>
              ) : null}
            </div>
            {description ? (
              <ShowMoreText
                previewLimit={80}
                variant="small"
                className="mt-4 !text-zinc-700"
              >
                {description}
              </ShowMoreText>
            ) : null}
          </div>

          <div className="flex flex-col gap-4">
            <Text variant="h4">You don’t have any runs</Text>
            <Text variant="large">
              Get started with creating a run, and you’ll see information here
            </Text>
          </div>
          <RunAgentModal
            triggerSlot={
              <Button variant="primary" size="large" className="w-full">
                <PlusIcon size={20} /> New Run
              </Button>
            }
            agent={agent}
            agentId={agent.id.toString()}
          />
        </div>
      </RunDetailCard>
    </div>
  );
}
