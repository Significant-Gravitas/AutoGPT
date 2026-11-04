import { useNowMinute } from "@/hooks/useNowMinute";
import type { Task } from "../../../../data/types";
import { cardClock, lastSteps } from "../../helpers";

export function useBoardCard(task: Task) {
  const live = task.state === "working" || task.state === "needs_you";
  useNowMinute(live);

  return {
    clock: cardClock(task, Date.now()),
    steps: lastSteps(task),
  };
}
