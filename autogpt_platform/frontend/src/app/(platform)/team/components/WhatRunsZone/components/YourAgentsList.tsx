import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { STATUS_BADGE_CLASS } from "../constants";
import { getAdoptableExperts, getAdoptTargetVersionID } from "../helpers";
import { AdoptAgentButton } from "./AdoptAgentButton";

type Props = {
  agents: LibraryAgent[];
  experts: Expert[];
  libraryAgentCount: number;
  pendingLibraryAgentIDs: Set<string>;
  adoptedTargetKeys: Set<string>;
  hasMoreAgents: boolean;
  isLoadingMoreAgents: boolean;
  isErrorLoadingMoreAgents: boolean;
  onLoadMore: () => void;
  onAdopt: (agent: LibraryAgent, expert: Expert) => void;
};

export function YourAgentsList({
  agents,
  experts,
  libraryAgentCount,
  pendingLibraryAgentIDs,
  adoptedTargetKeys,
  hasMoreAgents,
  isLoadingMoreAgents,
  isErrorLoadingMoreAgents,
  onLoadMore,
  onAdopt,
}: Props) {
  return (
    <section aria-label="Your agents" className="flex flex-col gap-2">
      <Text variant="small-medium" className="text-zinc-500">
        Your agents
      </Text>
      {agents.length === 0 ? (
        !hasMoreAgents ? (
          <Text variant="small" className="text-zinc-500">
            {libraryAgentCount === 0
              ? "No available agents to adopt."
              : "Every agent is already on your team."}
          </Text>
        ) : null
      ) : (
        <div className="divide-y divide-zinc-100 rounded-2xl border border-zinc-200 bg-white">
          {agents.map((agent) => {
            const canAdopt = Boolean(getAdoptTargetVersionID(agent));
            const adoptableExperts = getAdoptableExperts(
              agent,
              experts,
              adoptedTargetKeys,
            );
            return (
              <div
                key={agent.id}
                data-testid="what-runs-agent-row"
                className="flex items-center gap-3 px-4 py-3"
              >
                <ExpertAvatar
                  name={agent.name}
                  avatarUrl={agent.image_url ?? null}
                  size={32}
                />
                <div className="min-w-0 flex-1">
                  <Text variant="body" className="truncate">
                    {agent.name}
                  </Text>
                  <Text variant="small" className="text-zinc-500">
                    {agent.creator_name}
                  </Text>
                </div>
                {canAdopt ? (
                  <AdoptAgentButton
                    agent={agent}
                    experts={adoptableExperts}
                    isPending={pendingLibraryAgentIDs.has(agent.id)}
                    onAdopt={onAdopt}
                  />
                ) : (
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span
                        tabIndex={0}
                        className="shrink-0 rounded-full focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-zinc-900"
                      >
                        <Badge
                          variant="info"
                          className={`${STATUS_BADGE_CLASS} text-zinc-500`}
                        >
                          Local only
                        </Badge>
                      </span>
                    </TooltipTrigger>
                    <TooltipContent>
                      Publish this agent to the Marketplace before adopting it.
                    </TooltipContent>
                  </Tooltip>
                )}
              </div>
            );
          })}
        </div>
      )}
      {isErrorLoadingMoreAgents ? (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-200 px-3 py-2">
          <Text variant="small" className="text-zinc-500">
            We could not load more agents.
          </Text>
          <Button variant="secondary" size="small" onClick={onLoadMore}>
            Retry loading more
          </Button>
        </div>
      ) : hasMoreAgents ? (
        <Button
          variant="secondary"
          size="small"
          loading={isLoadingMoreAgents}
          onClick={onLoadMore}
          className="self-start"
        >
          Load more agents
        </Button>
      ) : null}
    </section>
  );
}
