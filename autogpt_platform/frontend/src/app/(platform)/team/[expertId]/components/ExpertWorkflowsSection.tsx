"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { getRaisedExpertAccent } from "@/app/(platform)/marketplace/components/ExpertsSection/helpers";
import { SearchInput } from "@/components/molecules/SearchInput/SearchInput";
import {
  GridViewIcon,
  ListViewIcon,
  PlusSignIcon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import {
  filterExpertWorkflows,
  WORKFLOW_FILTERS,
  WorkflowFilter,
} from "../../helpers";
import { ExpertWorkflowCard } from "./ExpertWorkflowCard";
import { ExpertWorkflowListItem } from "./ExpertWorkflowListItem";
import { FilterIconMenu } from "./FilterIconMenu";
import { useExpertWorkflowsView } from "./useExpertWorkflowsView";
import { ViewToggle } from "./ViewToggle";

const VIEW_OPTIONS = [
  { value: "list", label: "List view", icon: ListViewIcon },
  { value: "grid", label: "Grid view", icon: GridViewIcon },
] as const;

interface Props {
  expert: Expert;
  onInstallWorkflow: () => void;
}

export function ExpertWorkflowsSection({ expert, onInstallWorkflow }: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<WorkflowFilter>("all");
  const { view, setView } = useExpertWorkflowsView();
  const accent = getRaisedExpertAccent(expert.role, expert.color);
  const visible = filterExpertWorkflows(expert.workflows, query, filter);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="large-medium">{expert.name}&apos;s Workflows</Text>
        <div className="flex items-center gap-2">
          <Button
            variant="secondary"
            size="small"
            leftIcon={<Icon icon={PlusSignIcon} size={16} />}
            onClick={onInstallWorkflow}
          >
            Install workflow
          </Button>
          <SearchInput
            size="small"
            value={query}
            onChange={setQuery}
            placeholder="Search workflows"
            className="w-56"
          />
          <FilterIconMenu
            label="Filter workflows"
            value={filter}
            defaultValue="all"
            options={WORKFLOW_FILTERS}
            onChange={setFilter}
          />
          <ViewToggle value={view} options={VIEW_OPTIONS} onChange={setView} />
        </div>
      </div>
      {expert.workflows.length === 0 ? (
        <p className="text-sm text-zinc-500">
          No workflows yet. Install one to give {expert.name} something to run.
        </p>
      ) : visible.length === 0 ? (
        <p className="text-sm text-zinc-500">No workflows match.</p>
      ) : view === "list" ? (
        <div className="flex flex-col gap-3 pt-4" data-testid="workflow-list">
          {visible.map((workflow) => (
            <ExpertWorkflowListItem
              key={workflow.id}
              workflow={workflow}
              expertId={expert.id}
              accentClassName={accent.pill}
            />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-2 lg:grid-cols-3">
          {visible.map((workflow) => (
            <ExpertWorkflowCard
              key={workflow.id}
              workflow={workflow}
              expertId={expert.id}
              coverColor={expert.color}
            />
          ))}
        </div>
      )}
    </section>
  );
}
