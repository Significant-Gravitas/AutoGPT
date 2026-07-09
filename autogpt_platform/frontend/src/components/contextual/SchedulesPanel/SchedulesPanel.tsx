"use client";

import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { PlusIcon } from "@phosphor-icons/react";
import { NEW_SCHEDULED_TASK_PROMPT } from "../guidedPrompts";
import { EmptyFollowups } from "./components/EmptyFollowups/EmptyFollowups";
import { FollowupListItem } from "./components/FollowupListItem/FollowupListItem";
import { GraphScheduleListItem } from "./components/GraphScheduleListItem/GraphScheduleListItem";
import { useSchedulesPanel } from "./useSchedulesPanel";

interface Props {
  onGuidedPrompt: (prompt: string) => void;
  withHeading?: boolean;
}

export function SchedulesPanel({ onGuidedPrompt, withHeading = true }: Props) {
  const { schedules, isLoading, error, partialError } = useSchedulesPanel();

  return (
    <section className="space-y-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 flex-col gap-2">
          {withHeading && <Text variant="h2">Scheduled</Text>}
          <Text variant="body" className="!text-zinc-500">
            Every automated job in one place — follow-up messages your AutoPilot
            will send itself AND recurring agent runs from the builder. Open a
            row to jump into the session / agent, or cancel one you no longer
            need.
          </Text>
        </div>
        <div className="flex flex-shrink-0 items-center gap-2">
          <Button
            variant="primary"
            size="small"
            onClick={() => onGuidedPrompt(NEW_SCHEDULED_TASK_PROMPT)}
            data-testid="schedule-new-button"
          >
            <PlusIcon className="mr-1 h-4 w-4" />
            New scheduled task
          </Button>
        </div>
      </header>

      {partialError && (
        <Text
          variant="body"
          className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 !text-amber-700"
          data-testid="schedules-partial-error"
        >
          Some scheduled items couldn&apos;t be loaded — showing the ones that
          did. Refresh to try again.
        </Text>
      )}

      {error ? (
        <ErrorCard
          responseError={{
            message:
              error instanceof Error
                ? error.message
                : "Failed to load schedules",
          }}
          context="scheduled items"
        />
      ) : isLoading ? (
        <div
          className="flex items-center justify-center py-16"
          data-testid="followups-loading"
        >
          <LoadingSpinner />
        </div>
      ) : schedules.length === 0 ? (
        <EmptyFollowups />
      ) : (
        <ul
          className="flex flex-col gap-3"
          data-testid="followups-list"
          aria-label="Scheduled items"
        >
          {schedules.map((schedule) => (
            <li key={`${schedule.kind}:${schedule.item.id}`}>
              {schedule.kind === "copilot_turn" ? (
                <FollowupListItem followup={schedule.item} />
              ) : (
                <GraphScheduleListItem schedule={schedule.item} />
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
