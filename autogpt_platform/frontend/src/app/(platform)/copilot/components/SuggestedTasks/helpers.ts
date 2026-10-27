import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { SuggestedTaskState } from "@/app/api/__generated__/models/suggestedTaskState";
import { Key, storage } from "@/services/storage/local-storage";
import type { ExpertIdentity, ExpertIdentityMap } from "../../useExpertMap";
import {
  Analytics01Icon,
  CalendarCheckIcon,
  Chat01Icon,
  Clock01Icon,
  CodeIcon,
  DollarCircleIcon,
  File02Icon,
  FlashIcon,
  GlobeIcon,
  Mail01Icon,
  Megaphone01Icon,
  News01Icon,
  Notification02Icon,
  Robot01Icon,
  Rocket01Icon,
  Search01Icon,
  ShoppingCart01Icon,
  SparklesIcon,
  Target01Icon,
  UserGroupIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

// Slugs the backend may emit (see intro.PROMPT_ICONS); anything unknown
// falls back to the sparkle.
const PROMPT_ICONS: Record<string, IconSvgElement> = {
  sparkle: SparklesIcon,
  "chart-bar": Analytics01Icon,
  envelope: Mail01Icon,
  "magnifying-glass": Search01Icon,
  "calendar-check": CalendarCheckIcon,
  bell: Notification02Icon,
  "rocket-launch": Rocket01Icon,
  "file-text": File02Icon,
  globe: GlobeIcon,
  code: CodeIcon,
  newspaper: News01Icon,
  users: UserGroupIcon,
  "shopping-cart": ShoppingCart01Icon,
  chats: Chat01Icon,
  lightning: FlashIcon,
  target: Target01Icon,
  robot: Robot01Icon,
  clock: Clock01Icon,
  megaphone: Megaphone01Icon,
  "currency-dollar": DollarCircleIcon,
};

export function getPromptIcon(slug: string | undefined) {
  return PROMPT_ICONS[slug ?? ""] ?? SparklesIcon;
}

export function getTaskState(task: SuggestedTask): SuggestedTaskState {
  return task.state ?? "open";
}

export function getShownTasks(tasks: SuggestedTask[]) {
  return tasks.filter((task) => getTaskState(task) !== "dismissed");
}

export function countOpenTasks(tasks: SuggestedTask[]) {
  return tasks.filter((task) => getTaskState(task) === "open").length;
}

export function getChatHref(sessionId: string) {
  return `/copilot?sessionId=${encodeURIComponent(sessionId)}`;
}

// The first three by stored (ranked) order are the plan; the rest sit under
// "{n} more tasks".
export const TOP_TASKS = 3;

export function splitTasks(tasks: SuggestedTask[]) {
  const top = tasks.slice(0, TOP_TASKS);
  return {
    top: getShownTasks(top),
    more: getShownTasks(tasks.slice(TOP_TASKS)),
    // Sent or removed one of the first three: the rest are what comes next.
    hasActedOnTop: top.some((task) => getTaskState(task) !== "open"),
  };
}

// The row that carries "Start here": the first open one, in shown order.
export function getStartHereID(tasks: SuggestedTask[]) {
  return tasks.find((task) => getTaskState(task) === "open")?.id ?? null;
}

// Each row's own control carries this, so focus can find a row by its task.
export const TASK_CONTROL_ATTRIBUTE = "data-task-control";

// Focuses a row's control once the list has re-rendered, or the element
// `fallbackID` names when there is no such row.
export function focusTaskRow(taskID: string | null, fallbackID?: string) {
  requestAnimationFrame(() => {
    const row = taskID
      ? document.querySelector<HTMLElement>(
          `[${TASK_CONTROL_ATTRIBUTE}="${taskID.replace(/["\\]/g, "\\$&")}"]`,
        )
      : null;
    const fallback = fallbackID ? document.getElementById(fallbackID) : null;
    (row ?? fallback)?.focus();
  });
}

// The row to focus when `task` leaves the shown rows: the next one, else the
// one before.
export function getNeighbourID(rows: SuggestedTask[], task: SuggestedTask) {
  const index = rows.findIndex((row) => row.id === task.id);
  if (index === -1) return null;
  return (rows[index + 1] ?? rows[index - 1])?.id ?? null;
}

export type TaskOwner =
  | { kind: "expert"; expert: ExpertIdentity }
  | { kind: "otto" };

// Who has the task. An open one goes to its owner's hire, or to Otto when it
// has no owner or that expert is not on the team (fired, archived, never
// hired). A started one is with whoever its chat is with, fired since or not.
export function getTaskOwner(
  task: SuggestedTask,
  experts: ExpertIdentity[],
  everyExpert?: ExpertIdentityMap,
): TaskOwner {
  if (!task.expert_id) return { kind: "otto" };
  const expert =
    (task.session_id ? everyExpert?.get(task.expert_id) : undefined) ??
    experts.find((candidate) => candidate.id === task.expert_id);
  return expert ? { kind: "expert", expert } : { kind: "otto" };
}

export function getOwnerName(owner: TaskOwner) {
  return owner.kind === "expert" ? owner.expert.name : "Otto";
}

// Collapsing the list is a per-user UI preference, kept as the list of user
// ids who collapsed it so two accounts on one browser keep their own.
export function readCollapsed(userID: string | null) {
  if (!userID) return false;
  return readCollapsedUsers().includes(userID);
}

export function writeCollapsed(userID: string | null, collapsed: boolean) {
  if (!userID) return;
  const others = readCollapsedUsers().filter((id) => id !== userID);
  storage.set(
    Key.HOME_SUGGESTED_TASKS_COLLAPSED,
    JSON.stringify(collapsed ? [...others, userID] : others),
  );
}

function readCollapsedUsers(): string[] {
  try {
    const parsed: unknown = JSON.parse(
      storage.get(Key.HOME_SUGGESTED_TASKS_COLLAPSED) ?? "[]",
    );
    return Array.isArray(parsed)
      ? parsed.filter((id): id is string => typeof id === "string")
      : [];
  } catch {
    return [];
  }
}

interface MergeArgs {
  tasks: SuggestedTask[];
  updated: SuggestedTask[];
  ids: string[];
}

// Only the rows a write named take the server's answer: a later change still
// queued behind it keeps its optimistic state instead of flickering back.
export function mergeUpdatedTasks({ tasks, updated, ids }: MergeArgs) {
  const byId = new Map(updated.map((task) => [task.id, task]));
  return tasks.map((task) =>
    ids.includes(task.id) ? (byId.get(task.id) ?? task) : task,
  );
}

interface WithStateArgs {
  tasks: SuggestedTask[];
  ids: string[];
  state: SuggestedTaskState;
  /** For a sent task: its chat, so the row links there at once, and who
   *  took it (null for Otto), so the row names them. */
  sent?: { sessionID: string; expertID: string | null };
}

export function withState({ tasks, ids, state, sent }: WithStateArgs) {
  return tasks.map((task) => {
    if (!ids.includes(task.id)) return task;
    if (!sent) return { ...task, state };
    return {
      ...task,
      state,
      session_id: sent.sessionID,
      expert_id: sent.expertID,
    };
  });
}

// Tool names a title may mention, mapped to their logo in
// `public/integrations`. Names that are also ordinary words (Monday, Medium,
// Teams, X) are left out or need their full product name, so a "Monday
// status report" never shows a Monday.com logo.
const MENTIONED_PROVIDERS: { pattern: RegExp; provider: string }[] = [
  { pattern: /\bgithub\b/i, provider: "github" },
  { pattern: /\bLinear\b/, provider: "linear" },
  { pattern: /\bslack\b/i, provider: "slack" },
  { pattern: /\bnotion\b/i, provider: "notion" },
  { pattern: /\bhubspot\b/i, provider: "hubspot" },
  { pattern: /\bairtable\b/i, provider: "airtable" },
  { pattern: /\bstripe\b/i, provider: "stripe" },
  { pattern: /\bdiscord\b/i, provider: "discord" },
  { pattern: /\btelegram\b/i, provider: "telegram" },
  { pattern: /\bwhatsapp\b/i, provider: "whatsapp" },
  { pattern: /\breddit\b/i, provider: "reddit" },
  { pattern: /\btwitter\b/i, provider: "twitter" },
  { pattern: /\bintercom\b/i, provider: "intercom" },
  { pattern: /\bcalendly\b/i, provider: "calendly" },
  { pattern: /\btodoist\b/i, provider: "todoist" },
  { pattern: /\bmonday\.com\b/i, provider: "monday" },
  { pattern: /\b(gmail|google)\b/i, provider: "google" },
  { pattern: /\bmicrosoft teams\b/i, provider: "teams" },
  { pattern: /\b(jira|confluence|atlassian)\b/i, provider: "atlassian" },
  { pattern: /\bwordpress\b/i, provider: "wordpress" },
  { pattern: /\bwebflow\b/i, provider: "webflow" },
  { pattern: /\bzapier\b/i, provider: "zapier" },
  { pattern: /\bposthog\b/i, provider: "posthog" },
  { pattern: /\bsentry\b/i, provider: "sentry" },
  { pattern: /\bpaypal\b/i, provider: "paypal" },
  { pattern: /\bmiro\b/i, provider: "miro" },
  { pattern: /\bcanva\b/i, provider: "canva" },
];

const MAX_MENTIONED_PROVIDERS = 3;

// The tools a title names, in the order it names them.
export function getMentionedProviders(title: string) {
  return MENTIONED_PROVIDERS.map(({ pattern, provider }) => ({
    provider,
    at: title.search(pattern),
  }))
    .filter(({ at }) => at >= 0)
    .sort((a, b) => a.at - b.at)
    .map(({ provider }) => provider)
    .slice(0, MAX_MENTIONED_PROVIDERS);
}
