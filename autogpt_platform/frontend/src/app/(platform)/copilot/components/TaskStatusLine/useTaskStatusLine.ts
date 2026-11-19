import { useEffect, useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { getGetV2GetSessionQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import { useWorkAreaStore } from "../../workArea";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetTask } from "@/app/api/__generated__/endpoints/tasks/tasks";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "@/components/molecules/AutopilotAvatar/helpers";
import { isQueued } from "@/app/(platform)/tasks/queue";
import { scheduleStateOf } from "@/app/(platform)/tasks/schedule";
import {
  isSettled,
  secondsBetween,
  taskLineStateOf,
  watchOf,
  type StartedRepeat,
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
  const workState = taskLineStateOf(status);
  const state =
    workState === "working" && task && isQueued(task) ? "queued" : workState;
  // Waiting for its start time, or missed it: the card offers Start now.
  const repeat: StartedRepeat | null = task
    ? task.repeat
      ? {
          label: task.repeat.label,
          paused: !!task.repeat.paused,
          watch: watchOf(task.repeat.watch),
        }
      : null
    : (started.repeat ?? null);
  const schedule = scheduleStateOf(
    task ?? {
      status,
      start_at: started.startAt ? new Date(started.startAt) : null,
      missed_start: started.missedStart,
      repeat,
    },
  );

  // A task that starts or settles on its own (at its start time, in the
  // background) brings an answer into this chat; re-read the chat so it
  // shows without a reload.
  const queryClient = useQueryClient();
  const chatSessionID = useWorkAreaStore((s) => s.chatSessionID);
  const lastStatus = useRef(status);
  useEffect(() => {
    if (lastStatus.current === status) return;
    lastStatus.current = status;
    if (!chatSessionID) return;
    void queryClient.invalidateQueries({
      queryKey: getGetV2GetSessionQueryKey(chatSessionID),
    });
  }, [status, chatSessionID, queryClient]);

  const ownerExpertID = task ? task.owner_expert_id : started.ownerExpertID;
  const expertsQuery = useListExperts({ query: { enabled: !!ownerExpertID } });
  const experts =
    expertsQuery.data?.status === 200 ? expertsQuery.data.data : [];
  const expert = experts.find((e) => e.id === ownerExpertID) ?? null;

  const now = useNow(state === "working" && !schedule);
  const startedAt = task?.started_at ?? started.startedAt;
  const seconds =
    state === "working"
      ? secondsBetween(startedAt, now)
      : state === "done"
        ? secondsBetween(startedAt, task?.completed_at)
        : null;

  return {
    state,
    schedule,
    repeat,
    startAt: task ? (task.start_at ?? null) : (started.startAt ?? null),
    // A task cancelled before it ever started says Cancelled, not Stopped.
    neverStarted: !startedAt,
    seconds,
    title: task?.title ?? started.title,
    // Where it is in the worker's own words, once it has said so.
    statusText: task?.status_text ?? null,
    /** Why a task that did not deliver failed, in the worker's words. */
    failureReason: task?.failure_reason ?? null,
    subtasks:
      task && (task.subtask_count ?? 0) > 0
        ? { done: task.subtasks_done ?? 0, total: task.subtask_count ?? 0 }
        : null,
    openAsks: task?.open_ask_count ?? 0,
    /** Events a watching task still has to run after this one. */
    eventsWaiting: task?.events_waiting ?? 0,
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
