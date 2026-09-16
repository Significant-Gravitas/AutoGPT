import { Expert } from "@/app/api/__generated__/models/expert";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { workflowSubtitle } from "@/components/molecules/InstallWorkflowPicker/helpers";
import { getAdoptableExperts } from "../helpers";
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
    <section aria-label="Your workflows" className="!mt-8 flex flex-col gap-2">
      <div className="mb-4 border-b border-zinc-100 pb-4">
        <Text variant="h4">Your workflows</Text>
      </div>
      {agents.length === 0 ? (
        !hasMoreAgents ? (
          <Text variant="small" className="text-zinc-500">
            {libraryAgentCount === 0
              ? "No workflows available to install."
              : "Every workflow is already on your team."}
          </Text>
        ) : null
      ) : (
        <div className="divide-y divide-zinc-100 rounded-2xl border border-zinc-200 bg-white">
          {agents.map((agent) => {
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
                  <Text variant="small" className="truncate text-zinc-500">
                    {workflowSubtitle(agent.description)}
                  </Text>
                </div>
                <AdoptAgentButton
                  agent={agent}
                  experts={adoptableExperts}
                  isPending={pendingLibraryAgentIDs.has(agent.id)}
                  onAdopt={onAdopt}
                />
              </div>
            );
          })}
        </div>
      )}
      {isErrorLoadingMoreAgents ? (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-zinc-200 px-3 py-2">
          <Text variant="small" className="text-zinc-500">
            We could not load more workflows.
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
          Load more workflows
        </Button>
      ) : null}
    </section>
  );
}
