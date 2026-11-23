"use client";

import type { TaskDraft } from "@/app/(platform)/tasks/components/NewTaskDialog/draft";
import { useGetBrainDumpSuggestedTasks } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { TaskAdd01Icon } from "@hugeicons/core-free-icons";
import { draftFromSuggestion, openSuggestions } from "../firstTasks";

interface Props {
  experts: Expert[];
  onCreate: (from?: TaskDraft) => void;
}

/** The board before the first task: what it is for, Create task, and the
 *  user's own suggested first tasks when there are any. */
export function FirstTasks({ experts, onCreate }: Props) {
  const suggested = useGetBrainDumpSuggestedTasks({
    query: {
      select: (res) =>
        res.status === 200 ? openSuggestions(res.data.tasks ?? []) : [],
    },
  });
  const suggestions = suggested.data ?? [];

  return (
    <section
      aria-labelledby="first-tasks-heading"
      className="flex flex-col items-center gap-3 rounded-tray bg-surface-tray px-6 py-10 text-center"
    >
      <Text as="h3" id="first-tasks-heading" variant="inbox-medium" tone="ink">
        No tasks yet
      </Text>
      <Text as="p" variant="inbox" tone="faint" className="max-w-md">
        Give Otto or an expert something to do, now or on a schedule. Each task
        shows here while it is worked on.
      </Text>
      <Button
        variant="primary"
        size="small"
        leadingIcon={TaskAdd01Icon}
        onClick={() => onCreate()}
      >
        Create task
      </Button>
      {suggestions.length > 0 ? (
        <div className="flex w-full max-w-md flex-col gap-2 pt-2">
          <Text as="p" variant="small" tone="faint">
            Or start with one of these
          </Text>
          <ul className="grid gap-1.5">
            {suggestions.map((suggestion) => (
              <li key={suggestion.id}>
                <Button
                  variant="secondary"
                  size="small"
                  className="w-full justify-start"
                  onClick={() =>
                    onCreate(draftFromSuggestion(suggestion, experts))
                  }
                >
                  <span className="truncate" dir="auto">
                    {suggestion.title}
                  </span>
                </Button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}
