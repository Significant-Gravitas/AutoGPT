import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getAdoptTargetVersionId } from "../helpers";
import { AdoptAgentButton } from "./AdoptAgentButton";

interface Props {
  agents: LibraryAgent[];
  experts: Expert[];
  libraryAgentCount: number;
  pendingLibraryAgentIds: Set<string>;
  onAdopt: (agent: LibraryAgent, expert: Expert) => void;
}

export function YourAgentsList({
  agents,
  experts,
  libraryAgentCount,
  pendingLibraryAgentIds,
  onAdopt,
}: Props) {
  return (
    <section aria-label="Your agents" className="flex flex-col gap-2">
      <Text variant="small-medium" className="text-zinc-500">
        Your agents
      </Text>
      {agents.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          {libraryAgentCount === 0
            ? "No agents in your library yet."
            : "Every agent is already on your team."}
        </Text>
      ) : (
        <div className="divide-y divide-zinc-100 rounded-2xl border border-zinc-200 bg-white">
          {agents.map((agent) => {
            const canAdopt = Boolean(getAdoptTargetVersionId(agent));
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
                    experts={experts}
                    isPending={pendingLibraryAgentIds.has(agent.id)}
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
                          className="normal-case tracking-normal text-zinc-500"
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
    </section>
  );
}
