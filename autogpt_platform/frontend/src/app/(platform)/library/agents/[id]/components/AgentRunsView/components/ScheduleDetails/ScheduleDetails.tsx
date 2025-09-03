"use client";

import React from "react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import moment from "moment";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { useScheduleDetails } from "./useScheduleDetails";

interface ScheduleDetailsProps {
  agent: LibraryAgent;
  scheduleId: string;
}

export function ScheduleDetails({ agent, scheduleId }: ScheduleDetailsProps) {
  const { schedule, isLoading, error } = useScheduleDetails(
    agent.graph_id,
    scheduleId,
  );

  if (error) {
    return (
      <ErrorCard
        responseError={
          error
            ? {
                message: String(
                  (error as unknown as { message?: string })?.message ||
                    "Failed to load schedule",
                ),
              }
            : undefined
        }
        context="schedule"
      />
    );
  }

  if (isLoading && !schedule) {
    return (
      <div className="flex flex-col gap-6">
        <div className="h-8 w-60 animate-pulse rounded bg-zinc-200" />
        <div className="h-6 w-40 animate-pulse rounded bg-zinc-200" />
        <div className="h-64 w-full animate-pulse rounded bg-zinc-100" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex w-full items-center justify-between">
        <div className="flex w-full flex-col gap-0">
          <div className="flex w-full items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Text variant="h3" className="!font-normal">
                {agent.name}
              </Text>
            </div>
          </div>
          {schedule && (
            <Text variant="small" className="mt-1 !text-zinc-600">
              Next run {moment(schedule.next_run_time).fromNow()} | version{" "}
              {schedule.graph_version}
            </Text>
          )}
        </div>
      </div>

      <TabsLine defaultValue="input">
        <TabsLineList>
          <TabsLineTrigger value="input">Your input</TabsLineTrigger>
          <TabsLineTrigger value="schedule">Schedule</TabsLineTrigger>
        </TabsLineList>

        <TabsLineContent value="input">
          <div className="text-neutral-600">Coming soon</div>
        </TabsLineContent>

        <TabsLineContent value="schedule">
          {isLoading || !schedule ? (
            <div className="text-neutral-500">Loading…</div>
          ) : (
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium">Name</label>
                <p className="text-sm text-neutral-700">{schedule.name}</p>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium">Cron</label>
                <p className="text-sm text-neutral-700">{schedule.cron}</p>
              </div>
              <div className="flex flex-col gap-1.5">
                <label className="text-sm font-medium">Next run</label>
                <p className="text-sm text-neutral-700">
                  {moment(schedule.next_run_time).format("LLL")}
                </p>
              </div>
            </div>
          )}
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}
