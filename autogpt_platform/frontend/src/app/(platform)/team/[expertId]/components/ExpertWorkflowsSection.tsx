"use client";

import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import {
  Calendar03Icon,
  GridViewIcon,
  ListViewIcon,
  PlayIcon,
  PlusSignIcon,
  WorkflowSquare01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useState } from "react";
import {
  filterExpertWorkflows,
  WORKFLOW_FILTERS,
  WorkflowFilter,
} from "../../helpers";
import { ExpertWorkflowCard } from "./ExpertWorkflowCard";
import { ExpertWorkflowListItem } from "./ExpertWorkflowListItem";
import { useExpertWorkflowAgents } from "./useExpertWorkflowAgents";
import { useExpertWorkflowsView } from "./useExpertWorkflowsView";
import { ViewToggle } from "./ViewToggle";
import { SectionEmptyState } from "../../components/SectionEmptyState";
import { ToolbarSearch } from "../../components/ToolbarSearch";
import { PillFilterMenu } from "../../components/PillFilterMenu";

const VIEW_OPTIONS = [
  { value: "list", label: "List view", icon: ListViewIcon },
  { value: "grid", label: "Grid view", icon: GridViewIcon },
] as const;

interface Props {
  expertName: string;
  workflows: ExpertWorkflowRef[];
  expertId?: string;
  coverColor?: string;
  emptyMessage?: string;
  onInstallWorkflow?: () => void;
  onAskWorkflow?: (prompt: string) => void;
}

export function ExpertWorkflowsSection({
  expertName,
  workflows,
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
  const agentsByID = useExpertWorkflowAgents(workflows);

  return (
    <section>
      <div className="mb-2.5 flex flex-wrap items-center justify-between gap-3">
        <Text variant="large-medium" tone="primary">
          {expertName}&apos;s Workflows
        </Text>
        <div className="flex w-full flex-wrap items-center gap-2 sm:w-auto">
          {onInstallWorkflow ? (
            <Button
              variant="secondary"
              size="small"
              leadingIcon={PlusSignIcon}
              onClick={onInstallWorkflow}
            >
              Install workflow
            </Button>
          ) : null}
          <ToolbarSearch
            id="expert-workflows-search"
            label="Search workflows"
            value={query}
            onChange={setQuery}
          />
          <PillFilterMenu
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
        <SectionEmptyState
          tiles={[
            <Icon
              key="flow"
              icon={WorkflowSquare01Icon}
              size={22}
              className="text-sky-500"
            />,
            <Icon
              key="play"
              icon={PlayIcon}
              size={22}
              className="text-orange-500"
            />,
            <Icon
              key="calendar"
              icon={Calendar03Icon}
              size={22}
              className="text-violet-500"
            />,
          ]}
          title="No workflows yet"
          description={
            emptyMessage ??
            `Install a workflow to give ${expertName} something to run.`
          }
        >
          {onInstallWorkflow ? (
            <Button
              variant="primary"
              size="small"
              leadingIcon={PlusSignIcon}
              onClick={onInstallWorkflow}
            >
              Install workflow
            </Button>
          ) : null}
        </SectionEmptyState>
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
              libraryAgent={agentsByID.get(workflow.library_agent_id ?? "")}
              expertId={expertId}
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
              libraryAgent={agentsByID.get(workflow.library_agent_id ?? "")}
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
