import type { Task } from "@/app/api/__generated__/models/task";
import { useNowMinute } from "@/hooks/useNowMinute";
import { useState } from "react";
import { isSettled, statusAge } from "../../helpers";

/** What a row and a board card share: the ticking age and rename state. */
export function useTaskRow(task: Task) {
  const now = useNowMinute(!isSettled(task.status));
  const [isRenaming, setIsRenaming] = useState(false);

  return {
    age: statusAge(task, now),
    isRenaming,
    startRename: () => setIsRenaming(true),
    stopRename: () => setIsRenaming(false),
  };
}
