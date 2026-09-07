"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
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
  expertName: string;
  workflows: ExpertWorkflowRef[];
  accentClassName: string;
  expertId?: string;
  coverColor?: string;
  emptyMessage?: string;
  onInstallWorkflow?: () => void;
  onAskWorkflow?: (prompt: string) => void;
}

export function ExpertWorkflowsSection({
  expertName,
  workflows,
  accentClassName,
  expertId,
  coverColor,
  emptyMessage,
  onInstallWorkflow,
  onAskWorkflow,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<WorkflowFilter>("all");
  const { view, setView } = useExpertWorkflowsView();
  const visible = filterExpertWorkflows(workflows, query, filter);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="body-medium" tone="primary">
          {expertName}&apos;s Workflows
        </Text>
        <div className="flex items-center gap-2">
          {onInstallWorkflow ? (
            <Button
              variant="secondary"
              size="xs"
              leadingIcon={PlusSignIcon}
              onClick={onInstallWorkflow}
            >
              Install workflow
            </Button>
          ) : null}
          <SearchInput
            size="xsmall"
            value={query}
            onChange={setQuery}
            placeholder="Search workflows"
            className="w-48"
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
      {workflows.length === 0 ? (
        <Text variant="body" tone="muted">
          {emptyMessage ??
            `No workflows yet. Install one to give ${expertName} something to run.`}
        </Text>
      ) : visible.length === 0 ? (
        <Text variant="body" tone="muted">
          No workflows match.
        </Text>
      ) : view === "list" ? (
        <div className="flex flex-col gap-3 pt-4" data-testid="workflow-list">
          {visible.map((workflow) => (
            <ExpertWorkflowListItem
              key={workflow.id}
              workflow={workflow}
              expertId={expertId}
              accentClassName={accentClassName}
              onAsk={onAskWorkflow}
            />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 pt-4 sm:grid-cols-2 lg:grid-cols-3">
          {visible.map((workflow) => (
            <ExpertWorkflowCard
              key={workflow.id}
              workflow={workflow}
              expertId={expertId}
              coverColor={coverColor}
              onAsk={onAskWorkflow}
            />
          ))}
        </div>
      )}
    </section>
  );
}
