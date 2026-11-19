"use client";

import { ownerOf } from "@/app/(platform)/tasks/helpers";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { Task } from "@/app/api/__generated__/models/task";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  CARD,
  InfoMeta,
} from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";

interface Props {
  task: Task;
  experts: Expert[];
  isStarting: boolean;
  onStart: () => void;
  onSetAside: () => void;
  onOpen: () => void;
}

/** One task Otto drafted: who would do it, what and why now, with Start and
 *  Not now. Once started it stands for the running task and opens it. */
export function SuggestionCard({
  task,
  experts,
  isStarting,
  onStart,
  onSetAside,
  onOpen,
}: Props) {
  const owner = ownerOf(task, experts);
  const isStarted = !task.is_suggested;

  return (
    <li className={`${CARD} flex items-start gap-2.5 p-3`}>
      <ExpertAvatar name={owner.name} avatarUrl={owner.avatarUrl} size={26} />
      <div className="flex min-w-0 flex-1 flex-col gap-1">
        <Text variant="inbox" as="span" tone="ink" className="text-pretty">
          {task.title}
        </Text>
        <Text variant="small" as="span" tone="quiet" className="text-pretty">
          {isStarted
            ? `${owner.name} is on it.`
            : (task.suggested_reason ?? `${owner.name} could take this on.`)}
        </Text>
        <div className="mt-1.5 flex items-center gap-1.5">
          {isStarted ? (
            <>
              <InfoMeta>Started</InfoMeta>
              <Button variant="ghost-muted" size="xs" onClick={onOpen}>
                Open
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="secondary"
                size="xs"
                loading={isStarting}
                onClick={onStart}
                aria-label={`Start ${task.title}`}
              >
                Start
              </Button>
              <Button
                variant="ghost-muted"
                size="xs"
                disabled={isStarting}
                onClick={onSetAside}
                aria-label={`Not now: ${task.title}`}
              >
                Not now
              </Button>
            </>
          )}
        </div>
      </div>
    </li>
  );
}
