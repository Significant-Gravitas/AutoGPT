"use client";

import React from "react";
import {
  TabsLine,
  TabsLineList,
  TabsLineTrigger,
  TabsLineContent,
} from "@/components/molecules/TabsLine/TabsLine";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useRunsSidebar } from "./useRunsSidebar";
import { RunListItem } from "./components/RunListItem";
import { ScheduleListItem } from "./components/ScheduleListItem";
import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";

interface RunsSidebarProps {
  agent: LibraryAgent;
  selectedRunId?: string;
  onSelectRun: (id: string) => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    loading?: boolean;
  }) => void;
}

export function RunsSidebar({
  agent,
  selectedRunId,
  onSelectRun,
  onCountsChange,
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
  } = useRunsSidebar({
    graphId: agent.graph_id,
    onSelectRun,
    onCountsChange,
  });

  if (error) {
    return <ErrorCard responseError={error} />;
  }

  if (loading) {
    return (
      <div className="ml-6 w-[20vw] space-y-4">
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-32 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  return (
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
      className="min-w-0 overflow-hidden"
    >
      <TabsLineList>
        <TabsLineTrigger value="runs">
          Runs <span className="ml-3 inline-block">{runsCount}</span>
        </TabsLineTrigger>
        <TabsLineTrigger value="scheduled">
          Scheduled <span className="ml-3 inline-block">{schedulesCount}</span>
        </TabsLineTrigger>
      </TabsLineList>

      <>
        <TabsLineContent value="runs">
          <InfiniteList
            items={runs}
            hasMore={!!hasMoreRuns}
            isFetchingMore={isFetchingMoreRuns}
            onEndReached={fetchMoreRuns}
            className="flex flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 lg:flex-col lg:gap-3 lg:overflow-x-hidden"
            itemWrapperClassName="w-auto lg:w-full"
            renderItem={(run) => (
              <div className="w-[15rem] lg:w-full">
                <RunListItem
                  run={run}
                  title={agent.name}
                  selected={selectedRunId === run.id}
                  onClick={() => onSelectRun && onSelectRun(run.id)}
                />
              </div>
            )}
          />
        </TabsLineContent>
        <TabsLineContent value="scheduled">
          <div className="flex flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 lg:flex-col lg:gap-3 lg:overflow-x-hidden">
            {schedules.map((s: GraphExecutionJobInfo) => (
              <div className="w-[15rem] lg:w-full" key={s.id}>
                <ScheduleListItem
                  schedule={s}
                  selected={selectedRunId === `schedule:${s.id}`}
                  onClick={() => onSelectRun(`schedule:${s.id}`)}
                />
              </div>
            ))}
          </div>
        </TabsLineContent>
      </>
    </TabsLine>
  );
}
