"use client";

import { useEffect } from "react";
import Link from "next/link";
import { ArrowLeftIcon } from "@phosphor-icons/react";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { EmptyFollowups } from "./components/EmptyFollowups/EmptyFollowups";
import { FollowupListItem } from "./components/FollowupListItem/FollowupListItem";
import { GraphScheduleListItem } from "./components/GraphScheduleListItem/GraphScheduleListItem";
import { useFollowupsPage } from "./useFollowupsPage";

export default function FollowupsPage() {
  const { schedules, isLoading, error } = useFollowupsPage();

  useEffect(() => {
    document.title = "Scheduled – AutoGPT Platform";
  }, []);

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <Link
        href="/library"
        className="inline-flex items-center gap-1 text-sm text-zinc-500 hover:text-zinc-800"
        data-testid="followups-back-to-library"
      >
        <ArrowLeftIcon size={14} weight="bold" />
        Back to Library
      </Link>
      <header className="flex flex-col gap-2">
        <Text variant="h2">Scheduled</Text>
        <Text variant="body" className="!text-zinc-500">
          Every automated job in one place — follow-up messages your copilot
          will send itself AND recurring agent runs from the builder. Open a row
          to jump into the session / agent, or cancel one you no longer need.
        </Text>
      </header>

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
    </main>
  );
}
