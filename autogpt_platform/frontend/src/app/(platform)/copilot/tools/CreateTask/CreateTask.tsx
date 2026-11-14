"use client";

import {
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { useWorkAreaStore } from "../../workArea";
import type { CreatedTask } from "./helpers";

interface Props {
  task: CreatedTask;
}

/** A pill for a task Otto just created; Open shows it beside the chat. */
export function CreatedTaskCard({ task }: Props) {
  const openTask = useWorkAreaStore((s) => s.openTask);

  return (
    <div className="py-1.5">
      <div className="inline-flex max-w-full animate-inbox-enter items-center gap-2 rounded-full bg-surface py-1 pr-1 pl-2.5 shadow-(--inbox-shadow-card)">
        <Icon icon={CheckmarkCircle02Icon} size={14} tone="ok" aria-hidden />
        <Text
          variant="inbox"
          as="span"
          tone="text"
          className="min-w-0 truncate"
        >
          Created task:{" "}
          <Text variant="inbox-medium" as="span" tone="ink">
            {task.title}
          </Text>
        </Text>
        <button
          type="button"
          onClick={() => openTask(task.id, task.title)}
          aria-label={`Open task ${task.title}`}
          className="inline-flex shrink-0 items-center gap-1 rounded-full px-2 py-0.5 transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        >
          <Text variant="inbox-medium" as="span" tone="ink">
            Open
          </Text>
          <Icon icon={ArrowRight01Icon} size={12} tone="quiet" aria-hidden />
        </button>
      </div>
    </div>
  );
}
