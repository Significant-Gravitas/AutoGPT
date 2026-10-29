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
