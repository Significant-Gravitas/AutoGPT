"use client";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Tick02Icon } from "@hugeicons/core-free-icons";
import { TOPIC_NAMES } from "../helpers";

interface Props {
  covered: string[];
}

const TOPICS = ["work", "team", "week", "first_task", "tools"];

// The five things the interview asks about, ticked off as they come up.
export function CoverageRing({ covered }: Props) {
  const done = TOPICS.filter((topic) => covered.includes(topic)).length;
  return (
    <div
      role="group"
      aria-label={`${done} of ${TOPICS.length} topics covered`}
      className="flex w-full flex-wrap items-center justify-center gap-2"
    >
      {TOPICS.map((topic) => {
        const isDone = covered.includes(topic);
        return (
          <span
            key={topic}
            data-covered={isDone}
            className={cn(
              "flex items-center gap-1 rounded-full border px-2.5 py-1",
              isDone
                ? "border-brand/40 bg-accent-surface"
                : "border-border-base bg-surface",
            )}
          >
            {isDone && <Icon icon={Tick02Icon} size={12} aria-hidden />}
            <Text variant="small" tone={isDone ? undefined : "muted"}>
              {TOPIC_NAMES[topic]}
            </Text>
          </span>
        );
      })}
    </div>
  );
}
