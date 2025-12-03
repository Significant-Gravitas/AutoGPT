"use client";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { cn } from "@/lib/utils";
import { AGENT_LIBRARY_SECTION_PADDING_X } from "../../../helpers";
import { RunListItem } from "./components/RunListItem";
import { ScheduleListItem } from "./components/ScheduleListItem";
import { useSidebarRunsList } from "./useSidebarRunsList";

interface Props {
  agent: LibraryAgent;
  selectedRunId?: string;
  onSelectRun: (id: string, tab?: "runs" | "scheduled") => void;
  onClearSelectedRun?: () => void;
  onTabChange?: (tab: "runs" | "scheduled" | "templates") => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    loading?: boolean;
  }) => void;
}

export function SidebarRunsList({
  agent,
  selectedRunId,
  onSelectRun,
  onClearSelectedRun,
  onTabChange,
  onCountsChange,
}: Props) {
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
  } = useSidebarRunsList({
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
        const value = v as "runs" | "scheduled" | "templates";
        onTabChange?.(value);
        if (value === "runs") {
          if (runs && runs.length) {
            onSelectRun(runs[0].id, "runs");
          } else {
            onClearSelectedRun?.();
          }
        } else if (value === "scheduled") {
          if (schedules && schedules.length) {
            onSelectRun(schedules[0].id, "scheduled");
          } else {
            onClearSelectedRun?.();
          }
        } else if (value === "templates") {
          onClearSelectedRun?.();
        }
      }}
      className="flex min-h-0 flex-col overflow-hidden"
    >
      <TabsLineList className={AGENT_LIBRARY_SECTION_PADDING_X}>
        <TabsLineTrigger value="runs">
          Tasks <span className="ml-3 inline-block">{runsCount}</span>
        </TabsLineTrigger>
        <TabsLineTrigger value="scheduled">
          Scheduled <span className="ml-3 inline-block">{schedulesCount}</span>
        </TabsLineTrigger>
        <TabsLineTrigger value="templates">
          Templates <span className="ml-3 inline-block">0</span>
        </TabsLineTrigger>
      </TabsLineList>

      <>
        <TabsLineContent
          value="runs"
          className={cn(
            "flex min-h-0 flex-1 flex-col",
            AGENT_LIBRARY_SECTION_PADDING_X,
          )}
        >
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
                  onClick={() => onSelectRun && onSelectRun(run.id, "runs")}
                />
              </div>
            )}
          />
        </TabsLineContent>
        <TabsLineContent
          value="scheduled"
          className={cn(
            "mt-0 flex min-h-0 flex-1 flex-col",
            AGENT_LIBRARY_SECTION_PADDING_X,
          )}
        >
          <div className="flex h-full flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 lg:flex-col lg:gap-3 lg:overflow-x-hidden">
            {schedules.length > 0 ? (
              schedules.map((s: GraphExecutionJobInfo) => (
                <div className="w-[15rem] lg:w-full" key={s.id}>
                  <ScheduleListItem
                    schedule={s}
                    selected={selectedRunId === s.id}
                    onClick={() => onSelectRun(s.id, "scheduled")}
                  />
                </div>
              ))
            ) : (
              <div className="flex min-h-[50vh] flex-col items-center justify-center">
                <Text variant="large" className="text-zinc-700">
                  No scheduled agents
                </Text>
              </div>
            )}
          </div>
        </TabsLineContent>
        <TabsLineContent
          value="templates"
          className={cn(
            "mt-0 flex min-h-0 flex-1 flex-col",
            AGENT_LIBRARY_SECTION_PADDING_X,
          )}
        >
          <div className="flex h-full flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 lg:flex-col lg:gap-3 lg:overflow-x-hidden">
            <div className="flex min-h-[50vh] flex-col items-center justify-center">
              <Text variant="large" className="text-zinc-700">
                No templates saved
              </Text>
            </div>
          </div>
        </TabsLineContent>
      </>
    </TabsLine>
  );
}
