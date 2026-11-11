import { isSessionID } from "@/lib/api/chat";
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

export function greeting(name: string | null | undefined, now = new Date()) {
  const hour = now.getHours();
  const part = hour < 12 ? "morning" : hour < 18 ? "afternoon" : "evening";
  const first = name?.trim().split(/\s+/)[0];
  return first ? `Good ${part}, ${first}` : `Good ${part}`;
}

export function briefingLine(completedCount: number | undefined) {
  if (completedCount === 1) return "Your team finished 1 thing in the last day.";
  if (completedCount && completedCount > 1) {
    return `Your team finished ${completedCount} things in the last day.`;
  }
  return "What should we work on?";
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
