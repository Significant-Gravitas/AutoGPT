"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { isLargeScreen, useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useUserTimezone } from "@/lib/hooks/useUserTimezone";
import { formatInTimezone, getTimezoneDisplayName } from "@/lib/timezone-utils";
import { AgentInputsReadOnly } from "../../modals/AgentInputsReadOnly/AgentInputsReadOnly";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { SelectedScheduleActions } from "./components/SelectedScheduleActions";
import { useSelectedScheduleView } from "./useSelectedScheduleView";

interface Props {
  agent: LibraryAgent;
  scheduleId: string;
  onClearSelectedRun?: () => void;
  banner?: React.ReactNode;
  onSelectSettings?: () => void;
  selectedSettings?: boolean;
}

export function SelectedScheduleView({
  agent,
  scheduleId,
  onClearSelectedRun,
  banner,
  onSelectSettings,
  selectedSettings,
}: Props) {
  const { schedule, isLoading, error } = useSelectedScheduleView(
    agent.graph_id,
    scheduleId,
  );

  const userTimezone = useUserTimezone();

  const breakpoint = useBreakpoint();
  const isLgScreenUp = isLargeScreen(breakpoint);

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
        httpError={
          (error as any)?.status
            ? {
                status: (error as any).status,
                statusText: (error as any).statusText,
              }
            : undefined
        }
        context="schedule"
      />
    );
  }

  if (isLoading && !schedule) {
    return <LoadingSelectedContent agent={agent} />;
  }

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout
          agent={agent}
          banner={banner}
          onSelectSettings={onSelectSettings}
          selectedSettings={selectedSettings}
        >
          <div className="flex flex-col gap-4">
            <div className="flex w-full flex-col gap-0">
              <RunDetailHeader
                agent={agent}
                run={undefined}
                scheduleRecurrence={
                  schedule
                    ? `${humanizeCronExpression(schedule.cron || "")} · ${getTimezoneDisplayName(schedule.timezone || userTimezone || "UTC")}`
                    : undefined
                }
              />
              {schedule && !isLgScreenUp ? (
                <div className="mt-4">
                  <SelectedScheduleActions
                    agent={agent}
                    scheduleId={schedule.id}
                    onDeleted={onClearSelectedRun}
                  />
                </div>
              ) : null}
            </div>

            {/* Schedule Section */}
            <div id="schedule" className="scroll-mt-4">
              <RunDetailCard title="Schedule">
                {isLoading || !schedule ? (
                  <div className="text-neutral-500">
                    <LoadingSpinner />
                  </div>
                ) : (
                  <div className="relative flex flex-col gap-8">
                    <div className="flex flex-col gap-1.5">
                      <Text variant="large-medium">Name</Text>
                      <Text variant="body">{schedule.name}</Text>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Text variant="large-medium">Recurrence</Text>
                      <Text variant="body" className="flex items-center gap-3">
                        {humanizeCronExpression(schedule.cron)}{" "}
                        <span className="text-zinc-500">•</span>{" "}
                        <span className="text-zinc-500">
                          {getTimezoneDisplayName(
                            schedule.timezone || userTimezone || "UTC",
                          )}
                        </span>
                      </Text>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Text variant="large-medium">Next run</Text>
                      <Text variant="body" className="flex items-center gap-3">
                        {formatInTimezone(
                          schedule.next_run_time,
                          userTimezone || "UTC",
                          {
                            year: "numeric",
                            month: "long",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                            hour12: false,
                          },
                        )}{" "}
                        <span className="text-zinc-500">•</span>{" "}
                        <span className="text-zinc-500">
                          {getTimezoneDisplayName(
                            schedule.timezone || userTimezone || "UTC",
                          )}
                        </span>
                      </Text>
                    </div>
                  </div>
                )}
              </RunDetailCard>
            </div>

            {/* Input Section */}
            <div id="input" className="scroll-mt-4">
              <RunDetailCard title="Your input">
                <div className="relative">
                  <AgentInputsReadOnly
                    agent={agent}
                    inputs={schedule?.input_data}
                    credentialInputs={schedule?.input_credentials}
                  />
                </div>
              </RunDetailCard>
            </div>
          </div>
        </SelectedViewLayout>
      </div>
      {schedule && isLgScreenUp ? (
        <div className="max-w-[3.75rem] flex-shrink-0">
          <SelectedScheduleActions
            agent={agent}
            scheduleId={schedule.id}
            onDeleted={onClearSelectedRun}
          />
        </div>
      ) : null}
    </div>
  );
}
