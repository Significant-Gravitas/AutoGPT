import { useNowMinute } from "@/hooks/useNowMinute";
import { useState } from "react";
import { statusAge, statusLine } from "../../../../data/helpers";
import type { Task } from "../../../../data/types";

export function useBoardCard(task: Task) {
  const live = task.state === "working" || task.state === "needs_you";
  useNowMinute(live);
  const [isRenaming, setIsRenaming] = useState(false);

  return {
    line: statusLine(task),
    age: statusAge(task),
    isRenaming,
    startRename: () => setIsRenaming(true),
    stopRename: () => setIsRenaming(false),
  };
}
