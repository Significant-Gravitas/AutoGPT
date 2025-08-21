"use client";
import { Plus } from "lucide-react";
import React, { useEffect, useState } from "react";

import {
  GraphExecutionID,
  GraphExecutionMeta,
  LibraryAgent,
  LibraryAgentPreset,
  LibraryAgentPresetID,
  Schedule,
  ScheduleID,
} from "@/lib/autogpt-server-api";
import { cn } from "@/lib/utils";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/atoms/Button/Button";
import LoadingBox from "@/components/ui/loading";
import { InfiniteScroll } from "@/components/contextual/InfiniteScroll/InfiniteScroll";
import { Separator } from "@/components/ui/separator";

import { agentRunStatusMap } from "@/components/agents/agent-run-status-chip";
import AgentRunSummaryCard from "@/components/agents/agent-run-summary-card";
import { AgentRunsQuery } from "../../use-agent-runs";
import { ScrollArea } from "@/components/ui/scroll-area";

interface AgentRunsSelectorListProps {
  agent: LibraryAgent;
  agentRunsQuery: AgentRunsQuery;
  agentPresets: LibraryAgentPreset[];
  schedules: Schedule[];
  selectedView: { type: "run" | "preset" | "schedule"; id?: string };
  allowDraftNewRun?: boolean;
  onSelectRun: (id: GraphExecutionID) => void;
  onSelectPreset: (preset: LibraryAgentPresetID) => void;
  onSelectSchedule: (id: ScheduleID) => void;
  onSelectDraftNewRun: () => void;
  doDeleteRun: (id: GraphExecutionMeta) => void;
  doDeletePreset: (id: LibraryAgentPresetID) => void;
  doDeleteSchedule: (id: ScheduleID) => void;
  className?: string;
}

export function AgentRunsSelectorList({
  agent,
  agentRunsQuery: {
    agentRuns,
    agentRunsLoading,
    hasMoreRuns,
    fetchMoreRuns,
    isFetchingMoreRuns,
  },
  agentPresets,
  schedules,
  selectedView,
  allowDraftNewRun = true,
  onSelectRun,
  onSelectPreset,
  onSelectSchedule,
  onSelectDraftNewRun,
  doDeleteRun,
  doDeletePreset,
  doDeleteSchedule,
  className,
}: AgentRunsSelectorListProps): React.ReactElement {
  const [activeListTab, setActiveListTab] = useState<"runs" | "scheduled">(
    "runs",
  );

  useEffect(() => {
    if (selectedView.type === "schedule") {
      setActiveListTab("scheduled");
    } else {
      setActiveListTab("runs");
    }
  }, [selectedView]);

  const listItemClasses = "h-28 w-72 lg:w-full lg:h-32";

  return (
    <aside className={cn("flex flex-col gap-4", className)}>
      {allowDraftNewRun && (
        <Button
          className={"mb-4 hidden lg:flex"}
          onClick={onSelectDraftNewRun}
          leftIcon={<Plus className="h-6 w-6" />}
        >
          New {agent.has_external_trigger ? "trigger" : "run"}
        </Button>
      )}

      <div className="flex gap-2">
        <Badge
          variant={activeListTab === "runs" ? "secondary" : "outline"}
          className="cursor-pointer gap-2 rounded-full text-base"
          onClick={() => setActiveListTab("runs")}
        >
          <span>Runs</span>
          <span className="text-neutral-600">{agentRuns.length}</span>
        </Badge>

        <Badge
          variant={activeListTab === "scheduled" ? "secondary" : "outline"}
          className="cursor-pointer gap-2 rounded-full text-base"
          onClick={() => setActiveListTab("scheduled")}
        >
          <span>Scheduled</span>
          <span className="text-neutral-600">{schedules.length}</span>
        </Badge>
      </div>

      {/* Runs / Schedules list */}
      {agentRunsLoading && activeListTab === "runs" ? (
        <LoadingBox className="h-28 w-full lg:h-[calc(100vh-300px)] lg:w-72 xl:w-80" />
      ) : (
        <ScrollArea
          className="w-full lg:h-[calc(100vh-300px)] lg:w-72 xl:w-80"
          orientation={window.innerWidth >= 1024 ? "vertical" : "horizontal"}
        >
          <InfiniteScroll
            direction={window.innerWidth >= 1024 ? "vertical" : "horizontal"}
            hasNextPage={hasMoreRuns}
            fetchNextPage={fetchMoreRuns}
            isFetchingNextPage={isFetchingMoreRuns}
          >
            <div className="flex items-center gap-2 lg:flex-col">
              {/* New Run button - only in small layouts */}
              {allowDraftNewRun && (
                <Button
                  size="large"
                  className={
                    "flex h-12 w-40 items-center gap-2 py-6 lg:hidden " +
                    (selectedView.type == "run" && !selectedView.id
                      ? "agpt-card-selected text-accent"
                      : "")
                  }
                  onClick={onSelectDraftNewRun}
                  leftIcon={<Plus className="h-6 w-6" />}
                >
                  New {agent.has_external_trigger ? "trigger" : "run"}
                </Button>
              )}

              {activeListTab === "runs" ? (
                <>
                  {agentPresets
                    .toSorted(
                      (a, b) => b.updated_at.getTime() - a.updated_at.getTime(),
                    )
                    .map((preset) => (
                      <AgentRunSummaryCard
                        className={cn(listItemClasses, "lg:h-auto")}
                        key={preset.id}
                        type="preset"
                        status={preset.is_active ? "active" : "inactive"}
                        title={preset.name}
                        // timestamp={preset.last_run_time} // TODO: implement this
                        selected={selectedView.id === preset.id}
                        onClick={() => onSelectPreset(preset.id)}
                        onDelete={() => doDeletePreset(preset.id)}
                      />
                    ))}
                  {agentPresets.length > 0 && <Separator className="my-1" />}
                  {agentRuns
                    .toSorted(
                      (a, b) => b.started_at.getTime() - a.started_at.getTime(),
                    )
                    .map((run) => (
                      <AgentRunSummaryCard
                        className={listItemClasses}
                        key={run.id}
                        type="run"
                        status={agentRunStatusMap[run.status]}
                        title={
                          (run.preset_id
                            ? agentPresets.find((p) => p.id == run.preset_id)
                                ?.name
                            : null) ?? agent.name
                        }
                        timestamp={run.started_at}
                        selected={selectedView.id === run.id}
                        onClick={() => onSelectRun(run.id)}
                        onDelete={() => doDeleteRun(run)}
                      />
                    ))}
                </>
              ) : (
                schedules.map((schedule) => (
                  <AgentRunSummaryCard
                    className={listItemClasses}
                    key={schedule.id}
                    type="schedule"
                    status="scheduled" // TODO: implement active/inactive status for schedules
                    title={schedule.name}
                    timestamp={schedule.next_run_time}
                    selected={selectedView.id === schedule.id}
                    onClick={() => onSelectSchedule(schedule.id)}
                    onDelete={() => doDeleteSchedule(schedule.id)}
                  />
                ))
              )}
            </div>
          </InfiniteScroll>
        </ScrollArea>
      )}
    </aside>
  );
}
