import { useEffect, useState } from "react";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetTask } from "@/app/api/__generated__/endpoints/tasks/tasks";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import {
  isSettled,
  secondsBetween,
  taskLineStateOf,
  type StartedTask,
} from "./helpers";

const POLL_MS = 4_000;

/** Follows a started task from the status its tool result froze: the task is
 *  re-read until it settles, and the clock runs only while it is worked on. */
export function useTaskStatusLine(started: StartedTask) {
  const taskQuery = useGetTask(started.id, {
    query: {
      refetchInterval: (query) => {
        const response = query.state.data;
        const status =
          response?.status === 200 ? response.data.status : started.status;
        return isSettled(status) ? false : POLL_MS;
      },
    },
  });
  const task = taskQuery.data?.status === 200 ? taskQuery.data.data : null;
  const status = task?.status ?? started.status;
  const state = taskLineStateOf(status);

  const ownerExpertID = task ? task.owner_expert_id : started.ownerExpertID;
  const expertsQuery = useListExperts({ query: { enabled: !!ownerExpertID } });
  const experts =
    expertsQuery.data?.status === 200 ? expertsQuery.data.data : [];
  const expert = experts.find((e) => e.id === ownerExpertID) ?? null;

  const now = useNow(state === "working");
  const startedAt = task?.started_at ?? started.startedAt;
  const seconds =
    state === "working"
      ? secondsBetween(startedAt, now)
      : state === "done"
        ? secondsBetween(startedAt, task?.completed_at)
        : null;

  return {
    state,
    seconds,
    title: task?.title ?? started.title,
    // Where it is in the worker's own words, once it has said so.
    statusText: task?.status_text ?? null,
    subtasks:
      task && (task.subtask_count ?? 0) > 0
        ? { done: task.subtasks_done ?? 0, total: task.subtask_count ?? 0 }
        : null,
    openAsks: task?.open_ask_count ?? 0,
    owner: {
      name: expert?.name ?? (ownerExpertID ? "An expert" : AUTOPILOT_NAME),
      avatarUrl: ownerExpertID
        ? (expert?.avatar_url ?? null)
        : AUTOPILOT_AVATAR_URL,
    },
  };
}

function useNow(ticking: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!ticking) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [ticking]);
  return now;
}
