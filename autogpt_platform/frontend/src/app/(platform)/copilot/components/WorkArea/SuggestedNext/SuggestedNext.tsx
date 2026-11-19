"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { InfoGroup } from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { SuggestionCard } from "./SuggestionCard";
import { useSuggestedNext } from "./useSuggestedNext";

interface Props {
  experts: Expert[];
  onOpen: (task: Task) => void;
}

/** "Suggested next", near the top of the brief: the tasks Otto drafted, each
 *  started in one click. With none, only a quiet control to ask for some. */
export function SuggestedNext({ experts, onOpen }: Props) {
  const next = useSuggestedNext();
  const suggestMore = (
    <Button
      variant="ghost-muted"
      size="xs"
      loading={next.isSuggesting}
      onClick={() => void next.suggestMore()}
    >
      {next.isSuggesting
        ? "Drafting"
        : next.cards.length > 0
          ? "Suggest more"
          : "Suggest tasks"}
    </Button>
  );
  const notes = (
    <>
      {next.lastSetAside ? (
        <p className="flex items-center gap-1.5 px-1 text-body-sm text-foreground-quiet">
          <span className="min-w-0 truncate">
            Set aside {next.lastSetAside.title}.
          </span>
          <Button variant="link" size="inline-xs" onClick={next.undo}>
            Undo
          </Button>
        </p>
      ) : null}
      {next.isUpToDate ? (
        <Text variant="small" as="p" tone="quiet" className="mx-1">
          Nothing new to suggest right now.
        </Text>
      ) : null}
    </>
  );

  if (next.cards.length === 0) {
    return (
      <div className="flex flex-col items-end gap-1">
        {suggestMore}
        {notes}
      </div>
    );
  }

  return (
    <InfoGroup
      label="Suggested next"
      action={next.canSuggestMore ? suggestMore : undefined}
    >
      <ul className="flex flex-col gap-2">
        {next.cards.map((task) => (
          <SuggestionCard
            key={task.id}
            task={task}
            experts={experts}
            isStarting={next.startingID === task.id}
            onStart={() => next.start(task)}
            onSetAside={() => next.setAside(task)}
            onOpen={() => onOpen(task)}
          />
        ))}
      </ul>
      {notes}
    </InfoGroup>
  );
}
