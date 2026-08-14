import { Expert } from "@/app/api/__generated__/models/expert";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertWorkflowGroup } from "./components/ExpertWorkflowGroup";
import { WhatRunsFilters } from "./components/WhatRunsFilters";
import { YourAgentsList } from "./components/YourAgentsList";
import { useWhatRunsZone } from "./useWhatRunsZone";

interface Props {
  experts: Expert[];
}

export function WhatRunsZone({ experts }: Props) {
  const {
    filter,
    setFilter,
    groups,
    showAgents,
    unadoptedAgents,
    isLoadingAgents,
    adopt,
    pendingAgentId,
  } = useWhatRunsZone({ experts, enabled: experts.length > 0 });

  if (experts.length === 0) return null;

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
      ) : null}

      {showAgents ? (
        isLoadingAgents ? (
          <Skeleton className="h-24 w-full rounded-2xl" />
        ) : (
          <YourAgentsList
            agents={unadoptedAgents}
            experts={experts}
            pendingAgentId={pendingAgentId}
            onAdopt={adopt}
          />
        )
      ) : null}
    </section>
  );
}
