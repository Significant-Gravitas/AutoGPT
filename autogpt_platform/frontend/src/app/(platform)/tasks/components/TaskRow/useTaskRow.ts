import type { Task } from "@/app/api/__generated__/models/task";
import { useNowMinute } from "@/hooks/useNowMinute";
import { useState } from "react";
import { isSettled, statusAge } from "../../helpers";
import { isQueued } from "../../queue";
import { isScheduled, scheduleStateOf, waitingLine } from "../../schedule";
import { useStartCountdown } from "../../useStartCountdown";

/** What a row and a board card share: the ticking age, or how soon a
 *  scheduled task starts, and rename state. */
export function useTaskRow(task: Task) {
  const countdown = useStartCountdown(isScheduled(task) ? task.start_at : null);
  const now = useNowMinute(!countdown && !isSettled(task.status));
  const [isRenaming, setIsRenaming] = useState(false);

  const repeat = isScheduled(task) ? task.repeat : null;

  return {
    // Beside its Watching or Paused pill, a watching task says only what
    // it watches.
    age: repeat?.watch
      ? repeat.watch.line
      : repeat
        ? waitingLine(repeat, countdown?.label)
        : countdown
          ? countdown.label
          : statusAge(task, now),
    schedule: scheduleStateOf(task),
    queued: isQueued(task),
    isRenaming,
    startRename: () => setIsRenaming(true),
    stopRename: () => setIsRenaming(false),
  };
}
