"use client";

import { useEffect } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { EmptyFollowups } from "./components/EmptyFollowups/EmptyFollowups";
import { FollowupListItem } from "./components/FollowupListItem/FollowupListItem";
import { useFollowupsPage } from "./useFollowupsPage";

export default function FollowupsPage() {
  const { followups, isLoading, error, refetchFollowups } = useFollowupsPage();

  useEffect(() => {
    document.title = "Copilot follow-ups – AutoGPT Platform";
  }, []);

  return (
    <main className="container min-h-screen space-y-6 pb-20 pt-16 sm:px-8 md:px-12">
      <header className="flex flex-col gap-2">
        <Text variant="h2">Copilot follow-ups</Text>
        <Text variant="body" className="!text-zinc-500">
          Scheduled messages your copilot sessions will send themselves. Open a
          row to jump into the session, or cancel one you no longer need.
        </Text>
      </header>

      {error ? (
        <ErrorCard
          responseError={{
            message:
              error instanceof Error
                ? error.message
                : "Failed to load follow-ups",
          }}
          context="copilot follow-ups"
        />
      ) : isLoading ? (
        <div
          className="flex items-center justify-center py-16"
          data-testid="followups-loading"
        >
          <LoadingSpinner />
        </div>
      ) : followups.length === 0 ? (
        <EmptyFollowups />
      ) : (
        <ul
          className="flex flex-col gap-3"
          data-testid="followups-list"
          aria-label="Copilot follow-ups"
        >
          {followups.map((followup) => (
            <li key={followup.id}>
              <FollowupListItem
                followup={followup}
                onDeleted={refetchFollowups}
              />
            </li>
          ))}
        </ul>
      )}
    </main>
  );
}
