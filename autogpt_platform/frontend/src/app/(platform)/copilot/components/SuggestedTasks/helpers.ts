import type { SuggestedTask } from "@/app/api/__generated__/models/suggestedTask";
import type { SuggestedTaskState } from "@/app/api/__generated__/models/suggestedTaskState";
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
}

export function withState({ tasks, ids, state }: WithStateArgs) {
  return tasks.map((task) =>
    ids.includes(task.id) ? { ...task, state } : task,
  );
}
