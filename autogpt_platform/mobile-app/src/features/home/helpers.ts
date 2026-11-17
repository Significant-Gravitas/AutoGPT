import { isSessionID } from "@/lib/api/chat";
import type {
  HomeActiveTask,
  HomeAttentionItem,
  HomeDashboard,
  HomeUpcomingTask,
} from "@/lib/api/home";
import { FRONTEND_URL } from "@/lib/env";

export type HomeLinkTarget =
  | { kind: "chat"; sessionID: string }
  | { kind: "team" }
  | { kind: "web"; url: string };

const CHAT_HREF = /^\/copilot\?sessionId=([^&#]+)$/;

function decode(value: string) {
  try {
    return decodeURIComponent(value);
  } catch {
    return null;
  }
}

/** Where a home action's link goes. Only same-site paths are ever opened. */
export function homeLinkTarget(href: string): HomeLinkTarget | null {
  const chat = CHAT_HREF.exec(href)?.[1];
  const sessionID = chat ? decode(chat) : null;
  if (isSessionID(sessionID)) return { kind: "chat", sessionID };
  if (href === "/team") return { kind: "team" };
  if (href.startsWith("/") && !href.startsWith("//")) {
    return { kind: "web", url: FRONTEND_URL + href };
  }
  return null;
}

const TEAM_PLACEHOLDER =
  "What should the team work on? e.g. 'Follow up with yesterday's leads'";

/** The web's rule (`useHomeComposer`) for an account that already has work going. */
export function hasExistingWork(dashboard: HomeDashboard | undefined) {
  if (!dashboard) return false;
  return (
    dashboard.team.total > 0 ||
    dashboard.attention.length > 0 ||
    dashboard.active_tasks.length > 0 ||
    dashboard.upcoming_tasks.length > 0 ||
    dashboard.week.run_count > 0 ||
    (dashboard.recent_work?.total_count ?? 0) > 0 ||
    (dashboard.recent_work?.groups?.length ?? 0) > 0
  );
}

export function inputPlaceholder(
  expert: { name: string } | null,
  hasWork = false,
) {
  if (expert) return `What should ${expert.name} work on?`;
  return hasWork ? TEAM_PLACEHOLDER : "I'm a chef and I hate...";
}

const DAY_MS = 24 * 60 * 60 * 1000;

function startOfDay(date: Date) {
  return new Date(date.getFullYear(), date.getMonth(), date.getDate()).getTime();
}

/** `Today 16:00`, `Tomorrow 9:00`, `Monday 9:00`, or a date past a week. */
export function nextRunLabel(
  iso: string,
  now = new Date(),
  locale?: string,
) {
  const when = new Date(iso);
  const time = when.toLocaleTimeString(locale, {
    hour: "numeric",
    minute: "2-digit",
  });
  const days = Math.round((startOfDay(when) - startOfDay(now)) / DAY_MS);
  if (days <= 0) return `Today ${time}`;
  if (days === 1) return `Tomorrow ${time}`;
  if (days < 7) {
    return `${when.toLocaleDateString(locale, { weekday: "long" })} ${time}`;
  }
  return `${when.toLocaleDateString(locale, { day: "numeric", month: "short" })} ${time}`;
}

// The stored suggestions were written by a model asked about "AutoGPT
// features", so they name the product where the person is talking to Otto.
export function ottoWording(prompt: string) {
  return prompt.replace(/\bAutoGPT\b/g, "Otto");
}

export type HomeRow =
  | { kind: "attention"; item: HomeAttentionItem }
  | { kind: "active"; task: HomeActiveTask }
  | { kind: "upcoming"; task: HomeUpcomingTask }
  | { kind: "prompt"; prompt: string };

const MAX_ROWS = 4;
const MAX_STATUS_ROWS = 2;

interface HomeRowsInput {
  attention?: HomeAttentionItem[];
  activeTask?: HomeActiveTask | null;
  upcomingTask?: HomeUpcomingTask | null;
  prompts: string[];
}

/**
 * The short list above the composer: what needs you, then what is running
 * or next, then starters to fill the rest.
 */
export function homeRows({
  attention = [],
  activeTask,
  upcomingTask,
  prompts,
}: HomeRowsInput): HomeRow[] {
  const status: HomeRow[] = [
    ...attention.map((item) => ({ kind: "attention" as const, item })),
    ...(activeTask ? [{ kind: "active" as const, task: activeTask }] : []),
    ...(upcomingTask ? [{ kind: "upcoming" as const, task: upcomingTask }] : []),
  ].slice(0, MAX_STATUS_ROWS);
  const starters = prompts.map((prompt) => ({ kind: "prompt" as const, prompt }));
  return [...status, ...starters].slice(0, MAX_ROWS);
}
