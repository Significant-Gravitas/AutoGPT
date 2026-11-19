"use client";

import { asNextRun } from "@/app/(platform)/tasks/schedule";
import { useStartCountdown } from "@/app/(platform)/tasks/useStartCountdown";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskRepeat } from "@/app/api/__generated__/models/taskRepeat";
import { Icon } from "@/components/atoms/Icon/Icon";
import { WebhookURL } from "@/app/(platform)/tasks/components/WebhookURL/WebhookURL";
import { PauseIcon, RepeatIcon, FlashIcon } from "@hugeicons/core-free-icons";
import { InfoMeta } from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockRow } from "../blocks";

interface Props {
  task: Task;
  repeat: TaskRepeat;
}

/** The Details row of a repeating task: how often it runs, and when it runs
 *  next, or that it is paused and why. A task started by an event says what
 *  it watches and, for a webhook the user calls, the URL to call. */
export function RepeatRow({ task, repeat }: Props) {
  const isWaiting = task.status === "TODO";
  const watch = repeat.watch ?? null;
  const countdown = useStartCountdown(
    isWaiting && !repeat.paused && !watch ? task.start_at : null,
  );
  const subtitle = repeat.paused
    ? (task.status_text ?? "Paused")
    : countdown
      ? capitalise(asNextRun(countdown.label))
      : isWaiting
        ? (task.status_text ?? (watch ? "Watching" : null))
        : "Running now";

  return (
    <>
      <BlockRow
        tile={
          <Icon
            icon={repeat.paused ? PauseIcon : watch ? FlashIcon : RepeatIcon}
            size={14}
            tone={repeat.paused ? "quiet" : "info"}
          />
        }
        title={watch?.line ?? repeat.label}
        subtitle={subtitle}
        wrap
        trailing={countdown ? <InfoMeta>{countdown.exact}</InfoMeta> : null}
      />
      {watch?.url ? (
        <li className="px-2 pb-2">
          <WebhookURL url={watch.url} />
        </li>
      ) : null}
    </>
  );
}

function capitalise(text: string) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}
