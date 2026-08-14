import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getAdoptableListing } from "../helpers";
import { AdoptAgentButton } from "./AdoptAgentButton";

interface Props {
  agents: LibraryAgent[];
  experts: Expert[];
  hiddenAgentCount: number;
  pendingAgentIds: Set<string>;
  onAdopt: (agent: LibraryAgent, expert: Expert) => void;
}

export function YourAgentsList({
  agents,
  experts,
  hiddenAgentCount,
  pendingAgentIds,
  onAdopt,
}: Props) {
  return (
    <section aria-label="Your agents" className="flex flex-col gap-2">
      <Text variant="small-medium" className="text-zinc-500">
        Your agents
      </Text>
      {agents.length === 0 ? (
        <Text variant="small" className="text-zinc-500">
          Every agent is already on your team.
        </Text>
      ) : (
        <div className="divide-y divide-zinc-100 rounded-2xl border border-zinc-200 bg-white">
          {agents.map((agent) => {
            const canAdopt = getAdoptableListing(agent) !== null;
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
                    isPending={pendingAgentIds.has(agent.graph_id)}
                    onAdopt={onAdopt}
                  />
                ) : (
                  <span className="shrink-0 rounded-full bg-zinc-100 px-2.5 py-1 text-xs text-zinc-500">
                    Local only
                  </span>
                )}
              </div>
            );
          })}
        </div>
      )}
      {hiddenAgentCount > 0 ? (
        <Text variant="small" className="text-zinc-400">
          {hiddenAgentCount} more {hiddenAgentCount === 1 ? "agent" : "agents"}{" "}
          in your library {hiddenAgentCount === 1 ? "isn't" : "aren't"} shown
          here.
        </Text>
      ) : null}
    </section>
  );
}
