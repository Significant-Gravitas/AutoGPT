"use client";
import React, { useState } from "react";
import { Plus } from "lucide-react";

import { cn } from "@/lib/utils";
import {
  GraphExecutionID,
  GraphExecutionMeta,
  LibraryAgent,
  Schedule,
  ScheduleID,
} from "@/lib/autogpt-server-api";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/agptui/Button";
import { Badge } from "@/components/ui/badge";

import { agentRunStatusMap } from "@/components/agents/agent-run-status-chip";
import AgentRunSummaryCard from "@/components/agents/agent-run-summary-card";

interface AgentRunsSelectorListProps {
  agent: LibraryAgent;
  agentRuns: GraphExecutionMeta[];
  schedules: Schedule[];
  selectedView: { type: "run" | "schedule"; id?: string };
  onSelectRun: (id: GraphExecutionID) => void;
  onSelectSchedule: (schedule: Schedule) => void;
  onSelectDraftNewRun: () => void;
  onDeleteRun: (id: GraphExecutionMeta) => void;
  onDeleteSchedule: (id: ScheduleID) => void;
  className?: string;
}

export default function AgentRunsSelectorList({
  agent,
  agentRuns,
  schedules,
  selectedView,
  onSelectRun,
  onSelectSchedule,
  onSelectDraftNewRun,
  onDeleteRun,
  onDeleteSchedule,
  className,
}: AgentRunsSelectorListProps): React.ReactElement {
  const [activeListTab, setActiveListTab] = useState<"runs" | "scheduled">(
    "runs",
  );

  return (
    <aside className={cn("flex flex-col gap-4", className)}>
      <Button
        size="card"
        className={
          "mb-4 hidden h-16 w-72 items-center gap-2 py-6 lg:flex xl:w-80 " +
          (selectedView.type == "run" && !selectedView.id
            ? "agpt-card-selected text-accent"
            : "")
        }
        onClick={onSelectDraftNewRun}
      >
        <Plus className="h-6 w-6" />
        <span>New run</span>
      </Button>

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
          <span className="text-neutral-600">
            {schedules.filter((s) => s.graph_id === agent.agent_id).length}
          </span>
        </Badge>
      </div>

      {/* Runs / Schedules list */}
      <ScrollArea className="lg:h-[calc(100vh-200px)]">
        <div className="flex gap-2 lg:flex-col">
          {/* New Run button - only in small layouts */}
          <Button
            size="card"
            className={
              "flex h-28 w-40 items-center gap-2 py-6 lg:hidden " +
              (selectedView.type == "run" && !selectedView.id
                ? "agpt-card-selected text-accent"
                : "")
            }
            onClick={onSelectDraftNewRun}
          >
            <Plus className="h-6 w-6" />
            <span>New run</span>
          </Button>

          {activeListTab === "runs"
            ? agentRuns.map((run, i) => (
                <AgentRunSummaryCard
                  className="h-28 w-72 lg:h-32 xl:w-80"
                  key={i}
                  status={agentRunStatusMap[run.status]}
                  title={agent.name}
                  timestamp={run.started_at}
                  selected={selectedView.id === run.execution_id}
                  onClick={() => onSelectRun(run.execution_id)}
                  onDelete={() => onDeleteRun(run)}
                />
              ))
            : schedules
                .filter((schedule) => schedule.graph_id === agent.agent_id)
                .map((schedule, i) => (
                  <AgentRunSummaryCard
                    className="h-28 w-72 lg:h-32 xl:w-80"
                    key={i}
                    status="scheduled"
                    title={schedule.name}
                    timestamp={schedule.next_run_time}
                    selected={selectedView.id === schedule.id}
                    onClick={() => onSelectSchedule(schedule)}
                    onDelete={() => onDeleteSchedule(schedule.id)}
                  />
                ))}
        </div>
      </ScrollArea>
    </aside>
  );
}
