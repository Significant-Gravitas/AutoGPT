import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useStableCallback } from "@/hooks/useStableCallback";
import { markExpertThreadSource } from "@/services/experts/expert-thread-source";
import { trackFunnel } from "@/services/experts/experts-analytics";
import { useState } from "react";
import { toast as sonnerToast } from "sonner";
import { toastSendFailure } from "../../helpers/sendFailureToast";
import type { ExpertIdentity, ExpertIdentityMap } from "../../useExpertMap";
import {
  countOpenTasks,
  focusTaskRow,
  getShownTasks,
  getTaskOwner,
  getTaskState,
  type TaskOwner,
  withState,
} from "./helpers";
import { useSuggestedTaskList } from "./useSuggestedTaskList";
import { useTaskRecipient } from "./useTaskRecipient";

export type SuggestionSurface = "intro" | "home";

interface Args {
  surface: SuggestionSurface;
  /** Resolves to the chat the message went to, once it exists. */
  onSend: (message: string) => void | Promise<string | undefined | void>;
  /** False while the greeting is being written: the personalised list lands
   *  with it, and a read before then caches the generic set. */
  enabled: boolean;
  /** The user's active experts: who can take a task. */
  experts: ExpertIdentity[];
  /** Every expert, fired ones included: who a started task is with. */
  expertsById: ExpertIdentityMap;
  /** False until the roster has answered: before then a hired owner would
   *  read as gone and the task would go to Otto. */
  isRosterLoaded: boolean;
  /** The roster could not be read: owners are unknown, not gone. */
  hasRosterFailed: boolean;
  /** The composer's recipient, and how to change it: a task goes to its
   *  owner the way the recipient picker sends to an expert. */
  recipientID: string | null;
  selectRecipient: (id: string | null) => void;
  /** The page can start a task now (not resolving a recipient, not locked). */
  canSend: boolean;
}

const UNDO_TOAST_MS = 5000;

export function useSuggestedTasks({
  surface,
  onSend,
  enabled,
  experts,
  expertsById,
  isRosterLoaded,
  hasRosterFailed,
  recipientID,
  selectRecipient,
  canSend,
}: Args) {
  const [sendingId, setSendingId] = useState<string | null>(null);
  const recipient = useTaskRecipient({ recipientID, selectRecipient, canSend });

  const { list, tasks, mutate, updateCache } = useSuggestedTaskList({
    enabled,
  });
  const shownTasks = getShownTasks(tasks);

  // A task that did not start: the composer goes back to whoever it was
  // addressed to, and focus to the row, which is enabled again.
  function giveBack(task: SuggestedTask) {
    recipient.restore();
    focusTaskRow(task.id);
  }

  function sendTask(task: SuggestedTask) {
    if (sendingId || !isRosterLoaded) return;
    setSendingId(task.id);
    const owner = getTaskOwner(task, experts);
    trackFunnel("home_suggestion_clicked", {
      position: shownTasks.indexOf(task),
      surface,
      remaining: countOpenTasks(tasks) - 1,
      owner: owner.kind,
      expert_template_id: task.expert_template_id ?? null,
      rank: tasks.findIndex((candidate) => candidate.id === task.id),
    });
    recipient.sendTo(
      owner.kind === "expert" ? owner.expert.id : null,
      () => dispatch(task, owner),
      () => giveUp(task),
    );
  }

  function startChat(task: SuggestedTask, owner: TaskOwner) {
    markExpertThreadSource(owner.kind === "expert" ? "suggested_task" : null);
    return sendAndMark(task, owner)
      .then((isStarted) => {
        if (!isStarted) giveBack(task);
      })
      .catch((error: unknown) => {
        giveBack(task);
        toastSendFailure({
          title: "Could not start that task",
          error,
          recovery: "it is still in your suggestions",
        });
      })
      .finally(() => {
        markExpertThreadSource(null);
        setSendingId(null);
      });
  }

  async function sendAndMark(task: SuggestedTask, owner: TaskOwner) {
    const sessionId = await onSend(task.prompt);
    // No chat came of it (a locked composer, say): the row stays open.
    if (!sessionId) return false;
    updateCache((current) =>
      withState({
        tasks: current,
        ids: [task.id],
        state: "sent",
        sent: {
          sessionID: sessionId,
          expertID: owner.kind === "expert" ? owner.expert.id : null,
        },
      }),
    );
    mutate({
      data: { task_ids: [task.id], state: "sent", session_id: sessionId },
    });
    // Only an owner the roster says is gone; a roster that failed to load
    // says nothing about the team.
    if (owner.kind === "otto" && task.expert_template_id && !hasRosterFailed) {
      toast({
        title: `${task.expert_name ?? "That expert"} isn't on your team, so Otto has this one.`,
      });
    }
    return true;
  }

  function removeTask(task: SuggestedTask) {
    const before = tasks;
    trackFunnel("home_suggestion_dismissed", {
      position: shownTasks.indexOf(task),
      surface,
      remaining:
        countOpenTasks(tasks) - (getTaskState(task) === "open" ? 1 : 0),
      scope: "one",
    });
    updateCache((current) =>
      withState({ tasks: current, ids: [task.id], state: "dismissed" }),
    );
    mutate({ data: { task_ids: [task.id], state: "dismissed" } });

    const toastId = `suggestion-removed-${task.id}`;
    toast({
      title: "Task removed",
      duration: UNDO_TOAST_MS,
      replaceKey: toastId,
      action: (
        <Button
          variant="secondary"
          size="xs"
          onClick={() => {
            sonnerToast.dismiss(toastId);
            undoRemove(before, [task.id]);
          }}
        >
          Undo
        </Button>
      ),
    });
  }

  // An Undo sends no event: the dismissal stands as a fact.
  function undoRemove(before: SuggestedTask[], ids: string[]) {
    const previous = new Map(before.map((task) => [task.id, task]));
    updateCache((current) =>
      current.map((task) =>
        ids.includes(task.id) ? (previous.get(task.id) ?? task) : task,
      ),
    );
    mutate({ data: { task_ids: ids, state: "open" } });
  }

  // Runs the latest startChat (and so the latest onSend) whenever the
  // recipient switch it waited for lands.
  const dispatch = useStableCallback(
    (task: SuggestedTask, owner: TaskOwner) => void startChat(task, owner),
  );

  // The latest render's too: its restore sees the recipient the switch
  // landed on, which the click's render has not seen yet.
  const giveUp = useStableCallback((task: SuggestedTask) => {
    setSendingId(null);
    giveBack(task);
    toastSendFailure({
      title: "Could not start that task",
      error: new Error("The task's expert could not be selected"),
      recovery: "it is still in your suggestions",
    });
  });

  return {
    isStored: Boolean(list?.stored),
    /** Every task in stored (ranked) order, removed ones included. */
    allTasks: tasks,
    tasks: shownTasks,
    experts,
    expertsById,
    isRosterLoaded,
    sendingId,
    sendTask,
    removeTask,
  };
}
