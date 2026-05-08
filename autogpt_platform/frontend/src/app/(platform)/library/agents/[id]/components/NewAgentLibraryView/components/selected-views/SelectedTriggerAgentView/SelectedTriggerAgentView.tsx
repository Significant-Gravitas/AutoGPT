"use client";

import { useGetV2ListTriggerAgents } from "@/app/api/__generated__/endpoints/library/library";
import { useGetV1ListExecutionSchedulesForAGraph } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { okData } from "@/app/api/helpers";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { formatInTimezone, getTimezoneDisplayName } from "@/lib/timezone-utils";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { SelectedTriggerAgentActions } from "./SelectedTriggerAgentActions";

interface Props {
  agent: LibraryAgent;
  triggerAgentId: string;
  onClearSelectedRun?: () => void;
  banner?: React.ReactNode;
}

export function SelectedTriggerAgentView({
  agent,
  triggerAgentId,
  onClearSelectedRun,
  banner,
}: Props) {
  const { data, isLoading, error } = useGetV2ListTriggerAgents(agent.id, {
    query: { select: okData },
  });

  const triggerAgent = data?.find((t) => t.id === triggerAgentId);

  const {
    data: schedules,
    isLoading: isSchedulesLoading,
    error: schedulesError,
  } = useGetV1ListExecutionSchedulesForAGraph(triggerAgent?.graph_id || "", {
    query: {
      enabled: !!triggerAgent?.graph_id,
      select: okData,
    },
  });

  const rawUserTimezone = useUserTimezone();
  const displayTimezone =
    rawUserTimezone && rawUserTimezone !== "not-set"
      ? rawUserTimezone
      : Intl.DateTimeFormat().resolvedOptions().timeZone;
  const schedule = schedules?.[0];

  if (error || schedulesError) {
    const queryError = error ?? schedulesError;
    const message =
      queryError instanceof Error
        ? queryError.message
        : queryError != null
          ? String(queryError)
          : "Failed to load trigger data";
    return <ErrorCard responseError={{ message }} context="trigger agent" />;
  }

  if (isLoading || !data) {
    return <LoadingSelectedContent agent={agent} />;
  }

  if (!triggerAgent) {
    return (
      <SelectedViewLayout agent={agent} banner={banner}>
        <RunDetailCard title="Trigger not found">
          <Text variant="body" className="!text-zinc-500">
            This trigger agent no longer exists.
          </Text>
        </RunDetailCard>
      </SelectedViewLayout>
    );
  }

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout agent={agent} banner={banner}>
          <div className="flex flex-col gap-4">
            <RunDetailCard title={triggerAgent.name}>
              <div className="flex flex-col gap-3">
                {triggerAgent.description ? (
                  <Text variant="body">{triggerAgent.description}</Text>
                ) : (
                  <Text variant="body" className="!text-zinc-500">
                    No description.
                  </Text>
                )}
                <Text variant="small" className="!text-zinc-500">
                  Trigger agents run on a schedule and execute this agent when
                  their conditions are met. Edit them in the builder to change
                  what they do.
                </Text>
              </div>
            </RunDetailCard>

            {schedule && (
              <RunDetailCard title="Schedule">
                <div className="flex flex-col gap-6">
                  {schedule.graph_version !== triggerAgent.graph_version && (
                    <Text
                      variant="small"
                      className="rounded-md bg-amber-50 px-3 py-2 !text-amber-800"
                    >
                      This schedule is running version {schedule.graph_version}{" "}
                      of the trigger agent, but the latest version is{" "}
                      {triggerAgent.graph_version}. Ask AutoPilot to recreate
                      the schedule to pick up your latest edits.
                    </Text>
                  )}
                  <div className="flex flex-col gap-1.5">
                    <Text variant="large-medium">Recurrence</Text>
                    <Text variant="body" className="flex items-center gap-3">
                      {humanizeCronExpression(schedule.cron)}{" "}
                      <span className="text-zinc-500">&middot;</span>{" "}
                      <span className="text-zinc-500">
                        {getTimezoneDisplayName(
                          displayTimezone || schedule.timezone || "UTC",
                        )}
                      </span>
                    </Text>
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <Text variant="large-medium">Next run</Text>
                    <Text variant="body" className="flex items-center gap-3">
                      {formatInTimezone(
                        schedule.next_run_time,
                        displayTimezone || "UTC",
                        {
                          year: "numeric",
                          month: "long",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                          hour12: false,
                        },
                      )}{" "}
                      <span className="text-zinc-500">&middot;</span>{" "}
                      <span className="text-zinc-500">
                        {getTimezoneDisplayName(
                          displayTimezone || schedule.timezone || "UTC",
                        )}
                      </span>
                    </Text>
                  </div>
                </div>
              </RunDetailCard>
            )}

            {!schedule && !isSchedulesLoading && (
              <RunDetailCard title="Schedule">
                <Text variant="body" className="!text-zinc-500">
                  No schedule configured for this trigger agent.
                </Text>
              </RunDetailCard>
            )}
          </div>
        </SelectedViewLayout>
      </div>
      <div className="-mt-2 max-w-[3.75rem] flex-shrink-0">
        <SelectedTriggerAgentActions
          parentAgent={agent}
          triggerAgent={triggerAgent}
          onDeleted={onClearSelectedRun}
        />
      </div>
    </div>
  );
}
