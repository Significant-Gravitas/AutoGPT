"use client";

import type { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Skeleton } from "@/components/__legacy__/ui/skeleton";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { InfiniteList } from "@/components/molecules/InfiniteList/InfiniteList";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { RunListItem } from "./components/RunListItem";
import { ScheduleListItem } from "./components/ScheduleListItem";
import { TemplateListItem } from "./components/TemplateListItem";
import { useAgentRunsLists } from "./useAgentRunsLists";

interface Props {
  agent: LibraryAgent;
  selectedRunId?: string;
  onSelectRun: (id: string) => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    presetsCount: number;
    loading?: boolean;
  }) => void;
}

export function AgentRunsLists({
  agent,
  selectedRunId,
  onSelectRun,
  onCountsChange,
}: Props) {
  const {
    runs,
    schedules,
    presets,
    runsCount,
    schedulesCount,
    presetsCount,
    error,
    loading,
    hasMoreRuns,
    fetchMoreRuns,
    isFetchingMoreRuns,
    hasMorePresets,
    fetchMorePresets,
    isFetchingMorePresets,
    tabValue,
    setTabValue,
  } = useAgentRunsLists({
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
        setTabValue(value);
        if (value === "runs") {
          if (runs && runs.length) onSelectRun(runs[0].id);
        } else if (value === "scheduled") {
          if (schedules && schedules.length)
            onSelectRun(`schedule:${schedules[0].id}`);
        } else if (value === "templates") {
          if (presets && presets.length) onSelectRun(`preset:${presets[0].id}`);
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
        <TabsLineTrigger value="templates">
          {agent.trigger_setup_info ? "Triggers" : "Templates"}{" "}
          <span className="ml-3 inline-block">{presetsCount}</span>
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
        <TabsLineContent value="templates">
          <InfiniteList
            items={presets}
            hasMore={!!hasMorePresets}
            isFetchingMore={isFetchingMorePresets}
            onEndReached={fetchMorePresets}
            className="flex flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 lg:flex-col lg:gap-3 lg:overflow-x-hidden"
            itemWrapperClassName="w-auto lg:w-full"
            renderItem={(preset) => (
              <div className="w-[15rem] lg:w-full">
                <TemplateListItem
                  preset={preset}
                  selected={selectedRunId === `preset:${preset.id}`}
                  onClick={() =>
                    onSelectRun && onSelectRun(`preset:${preset.id}`)
                  }
                />
              </div>
            )}
          />
        </TabsLineContent>
      </>
    </TabsLine>
  );
}
