"use client";

import { useGetV1GetUserTimezone } from "@/app/api/__generated__/endpoints/auth/auth";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { humanizeCronExpression } from "@/lib/cron-expression-utils";
import { isLargeScreen, useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { formatInTimezone, getTimezoneDisplayName } from "@/lib/timezone-utils";
import { AgentInputsReadOnly } from "../../modals/AgentInputsReadOnly/AgentInputsReadOnly";
import { AnchorLinksWrap } from "../AnchorLinksWrap";
import { LoadingSelectedContent } from "../LoadingSelectedContent";
import { RunDetailCard } from "../RunDetailCard/RunDetailCard";
import { RunDetailHeader } from "../RunDetailHeader/RunDetailHeader";
import { SelectedViewLayout } from "../SelectedViewLayout";
import { SelectedScheduleActions } from "./components/SelectedScheduleActions";
import { useSelectedScheduleView } from "./useSelectedScheduleView";

const anchorStyles =
  "border-b-2 border-transparent pb-1 text-sm font-medium text-slate-600 transition-colors hover:text-slate-900 hover:border-slate-900";

interface Props {
  agent: LibraryAgent;
  scheduleId: string;
  onClearSelectedRun?: () => void;
}

export function SelectedScheduleView({
  agent,
  scheduleId,
  onClearSelectedRun,
}: Props) {
  const { schedule, isLoading, error } = useSelectedScheduleView(
    agent.graph_id,
    scheduleId,
  );

  const { data: userTzRes } = useGetV1GetUserTimezone({
    query: {
      select: (res) => (res.status === 200 ? res.data.timezone : undefined),
    },
  });

  const breakpoint = useBreakpoint();
  const isLgScreenUp = isLargeScreen(breakpoint);

  function scrollToSection(id: string) {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

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
    return <LoadingSelectedContent agentName={agent.name} agentId={agent.id} />;
  }

  return (
    <div className="flex h-full w-full gap-4">
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <SelectedViewLayout agentName={agent.name} agentId={agent.id}>
          <div className="flex flex-col gap-4">
            <div className="flex w-full flex-col gap-0">
              <RunDetailHeader
                agent={agent}
                run={undefined}
                scheduleRecurrence={
                  schedule
                    ? `${humanizeCronExpression(schedule.cron || "")} · ${getTimezoneDisplayName(schedule.timezone || userTzRes || "UTC")}`
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

            {/* Navigation Links */}
            <AnchorLinksWrap>
              <button
                onClick={() => scrollToSection("schedule")}
                className={anchorStyles}
              >
                Schedule
              </button>
              <button
                onClick={() => scrollToSection("input")}
                className={anchorStyles}
              >
                Your input
              </button>
            </AnchorLinksWrap>

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
                            schedule.timezone || userTzRes || "UTC",
                          )}
                        </span>
                      </Text>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Text variant="large-medium">Next run</Text>
                      <Text variant="body" className="flex items-center gap-3">
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
                        )}{" "}
                        <span className="text-zinc-500">•</span>{" "}
                        <span className="text-zinc-500">
                          {getTimezoneDisplayName(
                            schedule.timezone || userTzRes || "UTC",
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
