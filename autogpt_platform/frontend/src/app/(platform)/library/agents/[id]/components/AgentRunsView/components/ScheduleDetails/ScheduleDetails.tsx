"use client";

import React from "react";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Text } from "@/components/atoms/Text/Text";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";
import { useScheduleDetails } from "./useScheduleDetails";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { useGetV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";
import { formatInTimezone } from "@/lib/timezone-utils";
import { Skeleton } from "@/components/ui/skeleton";
import { AgentInputsReadOnly } from "../AgentInputsReadOnly/AgentInputsReadOnly";

interface ScheduleDetailsProps {
  agent: LibraryAgent;
  scheduleId: string;
}

export function ScheduleDetails({ agent, scheduleId }: ScheduleDetailsProps) {
  const { schedule, isLoading, error } = useScheduleDetails(
    agent.graph_id,
    scheduleId,
  );
  const { data: userTzRes } = useGetV1GetUserTimezone({
    query: {
      select: (res) => (res.status === 200 ? res.data.timezone : undefined),
    },
  });

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
      <div className="flex-1 space-y-4">
        <Skeleton className="h-8 w-full" />
        <Skeleton className="h-12 w-full" />
        <Skeleton className="h-64 w-full" />
        <Skeleton className="h-32 w-full" />
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      <RunDetailHeader agent={agent} run={undefined} />

      <TabsLine defaultValue="input">
        <TabsLineList>
          <TabsLineTrigger value="input">Your input</TabsLineTrigger>
          <TabsLineTrigger value="schedule">Schedule</TabsLineTrigger>
        </TabsLineList>

        <TabsLineContent value="input">
          <RunDetailCard>
            <div className="relative">
              {/*                 {// TODO: re-enable edit inputs modal once the API supports it */}
              {/* {schedule && Object.keys(schedule.input_data).length > 0 && (
                <EditInputsModal agent={agent} schedule={schedule} />
              )} */}
              <AgentInputsReadOnly
                agent={agent}
                inputs={schedule?.input_data}
              />
            </div>
          </RunDetailCard>
        </TabsLineContent>

        <TabsLineContent value="schedule">
          <RunDetailCard>
            {isLoading || !schedule ? (
              <div className="text-neutral-500">Loading…</div>
            ) : (
              <div className="relative flex flex-col gap-8">
                {
                  // TODO: re-enable edit schedule modal once the API supports it
                  /* <EditScheduleModal
                  graphId={agent.graph_id}
                  schedule={schedule}
                /> */
                }
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Name
                  </Text>
                  <p className="text-sm text-zinc-600">{schedule.name}</p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Recurrence
                  </Text>
                  <p className="text-sm text-zinc-600">
                    {humanizeCronExpression(schedule.cron, userTzRes)}
                  </p>
                </div>
                <div className="flex flex-col gap-1.5">
                  <Text variant="body-medium" className="!text-black">
                    Next run
                  </Text>
                  <p className="text-sm text-zinc-600">
                    {formatInTimezone(
                      schedule.next_run_time,
                      userTzRes || "UTC",
                      {
                        year: "numeric",
                        month: "long",
                        day: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                        hour12: false,
                      },
                    )}
                  </p>
                </div>
              </div>
            )}
          </RunDetailCard>
        </TabsLineContent>
      </TabsLine>
    </div>
  );
}
