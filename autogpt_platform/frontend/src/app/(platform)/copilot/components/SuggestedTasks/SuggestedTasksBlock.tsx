"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { SuggestedTaskList } from "./SuggestedTaskList";
import type { useSuggestedTasks } from "./useSuggestedTasks";

interface Props {
  suggestions: ReturnType<typeof useSuggestedTasks>;
  disabled?: boolean;
}

// The greeting's list after the greeting has retired: under the composer
// until every row is hidden.
export function SuggestedTasksBlock({ suggestions, disabled }: Props) {
  if (!suggestions.isStored || suggestions.tasks.length === 0) return null;

  return (
    <section
      className="group/block mx-auto mb-6 w-full max-w-[42rem] text-left"
      aria-label="Suggested first tasks"
    >
      <div className="mb-2 flex items-center justify-between px-1">
        <Text variant="eyebrow">Suggested first tasks</Text>
        <Button
          type="button"
          variant="ghost"
          size="xs"
          onClick={suggestions.hideAll}
          disabled={suggestions.sendingId !== null}
          className="text-zinc-400 opacity-0 transition-opacity duration-150 group-focus-within/block:opacity-100 group-hover/block:opacity-100 hover:text-zinc-700 focus-visible:opacity-100 [@media(hover:none)]:opacity-100"
        >
          Hide all
        </Button>
      </div>
      <SuggestedTaskList
        tasks={suggestions.tasks}
        sendingId={suggestions.sendingId}
        disabled={disabled}
        onSend={suggestions.sendTask}
        onHide={suggestions.hideTask}
      />
    </section>
  );
}
