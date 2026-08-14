import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertWorkflowGroup } from "./components/ExpertWorkflowGroup";
import { WhatRunsFilters } from "./components/WhatRunsFilters";
import { YourAgentsList } from "./components/YourAgentsList";
import { WhatRunsFilter } from "./helpers";
import { useWhatRunsZone } from "./useWhatRunsZone";

interface Props {
  experts: Expert[];
  schedules: GraphExecutionJobInfo[];
}

const GROUP_EMPTY_MESSAGES: Partial<Record<WhatRunsFilter, string>> = {
  workflows: "No workflows installed yet.",
  scheduled: "No scheduled workflows yet.",
};

export function WhatRunsZone({ experts, schedules }: Props) {
  const {
    filter,
    setFilter,
    groups,
    showAgents,
    unadoptedAgents,
    libraryAgentCount,
    isLoadingAgents,
    isErrorAgents,
    retryAgents,
    adopt,
    pendingLibraryAgentIds,
  } = useWhatRunsZone({
    experts,
    schedules,
    enabled: experts.length > 0,
  });

  if (experts.length === 0) return null;

  const groupEmptyMessage = GROUP_EMPTY_MESSAGES[filter];

  return (
    <section aria-label="What runs" className="flex flex-col gap-4">
      <div className="flex flex-col gap-1">
        <Text variant="h3">What runs</Text>
        <Text variant="body" className="max-w-prose text-zinc-600">
          Workflows installed on your experts, plus agents you can adopt.
        </Text>
      </div>

      <WhatRunsFilters value={filter} onChange={setFilter} />

      {groups.length > 0 ? (
        <div className="flex flex-col gap-3">
          {groups.map((group) => (
            <ExpertWorkflowGroup key={group.expert.id} group={group} />
          ))}
        </div>
      ) : groupEmptyMessage ? (
        <Text variant="small" className="text-zinc-500">
          {groupEmptyMessage}
        </Text>
      ) : null}

      {showAgents ? (
        isLoadingAgents ? (
          <Skeleton className="h-24 w-full rounded-2xl" />
        ) : isErrorAgents ? (
          <div className="flex items-center justify-between gap-3 rounded-2xl border border-zinc-200 bg-white px-4 py-3">
            <Text variant="small" className="text-zinc-500">
              We could not load your agents.
            </Text>
            <Button variant="secondary" size="small" onClick={retryAgents}>
              Retry
            </Button>
          </div>
        ) : (
          <YourAgentsList
            agents={unadoptedAgents}
            experts={experts}
            libraryAgentCount={libraryAgentCount}
            pendingLibraryAgentIds={pendingLibraryAgentIds}
            onAdopt={adopt}
          />
        )
      ) : null}
    </section>
  );
}
