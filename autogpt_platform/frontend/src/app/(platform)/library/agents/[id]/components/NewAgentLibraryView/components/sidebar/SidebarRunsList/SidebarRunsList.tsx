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
import { ScheduleListItem } from "./components/ScheduleListItem";
import { TaskListItem } from "./components/TaskListItem";
import { TemplateListItem } from "./components/TemplateListItem";
import { TriggerListItem } from "./components/TriggerListItem";
import { useSidebarRunsList } from "./useSidebarRunsList";

interface Props {
  agent: LibraryAgent;
  selectedRunId?: string;
  onSelectRun: (
    id: string,
    tab?: "runs" | "scheduled" | "templates" | "triggers",
  ) => void;
  onClearSelectedRun?: () => void;
  onTabChange?: (tab: "runs" | "scheduled" | "templates" | "triggers") => void;
  onCountsChange?: (info: {
    runsCount: number;
    schedulesCount: number;
    templatesCount: number;
    triggersCount: number;
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
    templates,
    triggers,
    runsCount,
    schedulesCount,
    templatesCount,
    triggersCount,
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
      <div
        className={cn(
          "ml-6 mt-8 w-[20vw] space-y-4",
          AGENT_LIBRARY_SECTION_PADDING_X,
        )}
      >
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
        const value = v as "runs" | "scheduled" | "templates" | "triggers";
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
        } else if (value === "triggers") {
          onClearSelectedRun?.();
        }
      }}
      className="flex min-h-0 flex-col overflow-hidden"
    >
      <div className="relative overflow-hidden">
        <div className="pointer-events-none absolute right-0 top-0 z-10 h-[46px] w-12 bg-gradient-to-l from-[#FAFAFA] to-transparent" />
        <div className="scrollbar-hide overflow-x-auto">
          <TabsLineList
            className={cn(AGENT_LIBRARY_SECTION_PADDING_X, "min-w-max")}
          >
            <TabsLineTrigger value="runs">
              Tasks <span className="ml-3 inline-block">{runsCount}</span>
            </TabsLineTrigger>
            <TabsLineTrigger value="scheduled">
              Scheduled{" "}
              <span className="ml-3 inline-block">{schedulesCount}</span>
            </TabsLineTrigger>
            {triggersCount > 0 && (
              <TabsLineTrigger value="triggers">
                Triggers{" "}
                <span className="ml-3 inline-block">{triggersCount}</span>
              </TabsLineTrigger>
            )}
            <TabsLineTrigger value="templates">
              Templates{" "}
              <span className="ml-3 inline-block">{templatesCount}</span>
            </TabsLineTrigger>
          </TabsLineList>
        </div>
      </div>

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
            className="flex max-h-[76vh] flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 lg:flex-col lg:gap-3 lg:overflow-y-auto lg:overflow-x-hidden"
            itemWrapperClassName="w-auto lg:w-full"
            renderItem={(run) => (
              <div className="w-[15rem] lg:w-full">
                <TaskListItem
                  run={run}
                  title={agent.name}
                  agent={agent}
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
          <div className="flex h-full flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 lg:flex-col lg:gap-3 lg:overflow-y-auto lg:overflow-x-hidden">
            {schedules.length > 0 ? (
              schedules.map((s: GraphExecutionJobInfo) => (
                <div className="w-[15rem] lg:w-full" key={s.id}>
                  <ScheduleListItem
                    schedule={s}
                    agent={agent}
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
        {triggersCount > 0 && (
          <TabsLineContent
            value="triggers"
            className={cn(
              "mt-0 flex min-h-0 flex-1 flex-col",
              AGENT_LIBRARY_SECTION_PADDING_X,
            )}
          >
            <div className="flex h-full flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 lg:flex-col lg:gap-3 lg:overflow-y-auto lg:overflow-x-hidden">
              {triggers.length > 0 ? (
                triggers.map((trigger) => (
                  <div className="w-[15rem] lg:w-full" key={trigger.id}>
                    <TriggerListItem
                      trigger={trigger}
                      agent={agent}
                      selected={selectedRunId === trigger.id}
                      onClick={() => onSelectRun(trigger.id, "triggers")}
                    />
                  </div>
                ))
              ) : (
                <div className="flex min-h-[50vh] flex-col items-center justify-center">
                  <Text variant="large" className="text-zinc-700">
                    No triggers set up
                  </Text>
                </div>
              )}
            </div>
          </TabsLineContent>
        )}
        <TabsLineContent
          value="templates"
          className={cn(
            "mt-0 flex min-h-0 flex-1 flex-col",
            AGENT_LIBRARY_SECTION_PADDING_X,
          )}
        >
          <div className="flex h-full flex-nowrap items-center justify-start gap-4 overflow-x-scroll px-1 pb-4 pt-1 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300 lg:flex-col lg:gap-3 lg:overflow-y-auto lg:overflow-x-hidden">
            {templates.length > 0 ? (
              templates.map((template) => (
                <div className="w-[15rem] lg:w-full" key={template.id}>
                  <TemplateListItem
                    template={template}
                    agent={agent}
                    selected={selectedRunId === template.id}
                    onClick={() => onSelectRun(template.id, "templates")}
                  />
                </div>
              ))
            ) : (
              <div className="flex min-h-[50vh] flex-col items-center justify-center">
                <Text variant="large" className="text-zinc-700">
                  No templates saved
                </Text>
              </div>
            )}
          </div>
        </TabsLineContent>
      </>
    </TabsLine>
  );
}
