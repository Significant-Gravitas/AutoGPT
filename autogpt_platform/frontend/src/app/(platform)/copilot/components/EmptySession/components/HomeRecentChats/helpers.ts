import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { Task } from "@/app/api/__generated__/models/task";

const UNTITLED = "Untitled chat";
/** Deeper than any worker chain: past it the chat counts as having no
 *  main chat. */
const MAX_PARENT_HOPS = 16;

/** Whether a chat is mid-turn, and the word a screen reader says for it. */
export function chatStatus(session: SessionSummaryResponse) {
  const isWorking = session.is_processing || session.chat_status === "queued";
  return { isWorking, label: isWorking ? "Working" : "Done" };
}

export function chatTitle(session: SessionSummaryResponse) {
  return session.title || UNTITLED;
}

/** The recipient's chats; a null expert is Otto. */
export function recentChatsFor(
  sessions: SessionSummaryResponse[],
  expertID: string | null,
) {
  return sessions.filter((session) => (session.expert_id ?? null) === expertID);
}

export interface TaskChat {
  session: SessionSummaryResponse;
  /** The task's title, not the brief the chat opened on. */
  title: string;
  /** The task it works on, when the task list holds it. */
  task: Task | null;
}

export interface ChatRow {
  session: SessionSummaryResponse;
  taskChats: TaskChat[];
}

/** A chat opened to do a task's work: a background worker, or the chat a
 *  task's Run opened. */
export function isTaskChat(session: SessionSummaryResponse) {
  return Boolean(session.parent_session_id || session.work_task_id);
}

/** The main chats, newest first as given, each with the task chats under it.
 *  A task chat whose main chat is not in the list is left out. */
export function buildChatRows(
  sessions: SessionSummaryResponse[],
  tasks: Task[],
): ChatRow[] {
  const mainChats = sessions.filter((session) => !isTaskChat(session));
  const mainIDs = new Set(mainChats.map((session) => session.id));
  const byID = new Map(sessions.map((session) => [session.id, session]));
  const placed = sessions.filter(isTaskChat).map((session) => ({
    parentID: mainChatOf(session, { byID, mainIDs, tasks }),
    chat: taskChat(session, tasks),
  }));
  return mainChats.map((session) => ({
    session,
    taskChats: placed
      .filter((entry) => entry.parentID === session.id)
      .map((entry) => entry.chat),
  }));
}

interface Lookup {
  byID: Map<string, SessionSummaryResponse>;
  mainIDs: Set<string>;
  tasks: Task[];
}

/** Walks up from a task chat to the main chat it was started from. */
function mainChatOf(session: SessionSummaryResponse, lookup: Lookup) {
  let current = session;
  for (let hop = 0; hop < MAX_PARENT_HOPS; hop++) {
    const parentID = parentIDOf(current, lookup);
    if (!parentID) return null;
    if (lookup.mainIDs.has(parentID)) return parentID;
    const parent = lookup.byID.get(parentID);
    if (!parent) return null;
    current = parent;
  }
  return null;
}

/** A worker names its parent; a Run chat's parent is the chat its task
 *  came from, one of the task's other chats. */
function parentIDOf(session: SessionSummaryResponse, lookup: Lookup) {
  if (session.parent_session_id) return session.parent_session_id;
  const others = (taskOf(session, lookup.tasks)?.session_ids ?? []).filter(
    (id) => id !== session.id,
  );
  return (
    others.find((id) => lookup.mainIDs.has(id)) ??
    others.find((id) => lookup.byID.has(id)) ??
    null
  );
}

function taskChat(session: SessionSummaryResponse, tasks: Task[]): TaskChat {
  const task = taskOf(session, tasks) ?? null;
  return { session, title: task?.title || chatTitle(session), task };
}

/** The task a chat works on. A worker is linked to its task beside the chat
 *  that started it; a task the worker started itself lacks that chat. */
function taskOf(session: SessionSummaryResponse, tasks: Task[]) {
  const parentID = session.parent_session_id;
  return (
    tasks.find((task) => task.id === session.work_task_id) ??
    tasks.find((task) => task.work_session_id === session.id) ??
    tasks.find((task) => {
      const ids = task.session_ids ?? [];
      return !!parentID && ids.includes(session.id) && ids.includes(parentID);
    })
  );
}
