"use client";

import React from "react";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
  TabsLineContent,
} from "@/components/molecules/TabsLine/TabsLine";
import { Button } from "@/components/atoms/Button/Button";
import { PlusIcon } from "@phosphor-icons/react/dist/ssr";
import { RunAgentModal } from "../RunAgentModal/RunAgentModal";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useRunsSidebar } from "./useRunsSidebar";
import { RunListItem } from "./components/RunListItem";
import { ScheduleListItem } from "./components/ScheduleListItem";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/ui/skeleton";

interface RunsSidebarProps {
  agent: LibraryAgent;
  selectedRunId?: string;
  onSelectRun: (id: string) => void;
}

export function RunsSidebar({
  agent,
  selectedRunId,
  onSelectRun,
}: RunsSidebarProps) {
  const {
    runs,
    schedules,
    runsCount,
    schedulesCount,
    error,
    loading,
    fetchMoreRuns,
    hasMoreRuns,
    isFetchingMoreRuns,
    tabValue,
    setTabValue,
  } = useRunsSidebar({ graphId: agent.graph_id, onSelectRun });

  if (error) {
    return <ErrorCard responseError={error} />;
  }

  if (loading) {
    return (
      <div className="ml-6 w-80 space-y-4">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  return (
    <div className="bg-gray-50 p-4 pl-5">
      <RunAgentModal
        triggerSlot={
          <Button variant="primary" size="large" className="w-full">
            <PlusIcon size={20} /> New Run
          </Button>
        }
        agent={agent}
        agentId={agent.id.toString()}
      />

      <TabsLine
        value={tabValue}
        onValueChange={(v) => {
          const value = v as "runs" | "scheduled";
          setTabValue(value);
          if (value === "runs") {
            if (runs && runs.length) onSelectRun(runs[0].id);
          } else {
            if (schedules && schedules.length)
              onSelectRun(`schedule:${schedules[0].id}`);
          }
        }}
        className="mt-6 overflow-x-auto"
      >
        <TabsLineList>
          <TabsLineTrigger value="runs">
            Runs <span className="ml-3 inline-block">{runsCount}</span>
          </TabsLineTrigger>
          <TabsLineTrigger value="scheduled">
            Scheduled{" "}
            <span className="ml-3 inline-block">{schedulesCount}</span>
          </TabsLineTrigger>
        </TabsLineList>

        <div className="px-[2px]">
          <TabsLineContent value="runs">
            <div className="-mx-1 flex gap-3 overflow-x-auto pb-1 [scrollbar-gutter:stable]">
              <div className="flex min-w-full gap-3 sm:min-w-0">
                <InfiniteList
                  items={runs}
                  hasMore={!!hasMoreRuns}
                  isFetchingMore={isFetchingMoreRuns}
                  onEndReached={fetchMoreRuns}
                  renderItem={(run) => (
                    <div className="mb-3 min-w-[320px] max-w-[360px] flex-shrink-0">
                      <RunListItem
                        run={run}
                        title={agent.name}
                        selected={selectedRunId === run.id}
                        onClick={() => onSelectRun && onSelectRun(run.id)}
                      />
                    </div>
                  )}
                />
              </div>
            </div>
          </TabsLineContent>
          <TabsLineContent value="scheduled">
            <div className="-mx-1 flex gap-3 overflow-x-auto pb-1 [scrollbar-gutter:stable]">
              {schedules.map((s: GraphExecutionJobInfo) => (
                <div
                  className="mb-3 min-w-[320px] max-w-[360px] flex-shrink-0"
                  key={s.id}
                >
                  <ScheduleListItem
                    schedule={s}
                    selected={selectedRunId === `schedule:${s.id}`}
                    onClick={() => onSelectRun(`schedule:${s.id}`)}
                  />
                </div>
              ))}
            </div>
          </TabsLineContent>
        </div>
      </TabsLine>
    </div>
  );
}
