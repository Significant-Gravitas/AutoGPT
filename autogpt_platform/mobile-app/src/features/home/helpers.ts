import { isSessionID } from "@/lib/api/chat";
import { FRONTEND_URL } from "@/lib/env";

import { expertRoleLabel } from "./expertRoleLabel";

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

/**
 * The first name to greet. Email sign-up stores the address's local part as
 * the name; that one was never given, so it does not count.
 */
export function firstName(
  name: string | null | undefined,
  email?: string | null,
) {
  const given = name?.trim();
  if (!given) return null;
  const emailName = email?.split("@")[0].toLowerCase();
  if (given.toLowerCase() === emailName) return null;
  return given.split(/\s+/)[0];
}

export function greetingLine(name: string | null) {
  return name ? `Hey, ${name}` : "Hey there";
}

interface Recipient {
  name: string;
  role: string | null;
}

/** The hero sentence before and after the picker; `null` is Otto. */
export function introParts(expert: Recipient | null) {
  if (!expert) {
    return {
      before: "Tell",
      after: " about your work, and it will find what to automate.",
    };
  }
  const role = expert.role?.trim();
  return {
    before: "I'm",
    after: role
      ? `, your ${expertRoleLabel(role)}. What should I take on?`
      : ". What should I take on?",
  };
}

/** A sentence part as words, with a leading comma or full stop kept apart to sit against the picker. */
export function sentenceWords(part: string) {
  const mark = /^[,.]/.exec(part)?.[0] ?? "";
  const words = part.slice(mark.length).trim().split(/\s+/).filter(Boolean);
  return { mark, words };
}

export function inputPlaceholder(expert: Pick<Recipient, "name"> | null) {
  return expert
    ? `What should ${expert.name} work on?`
    : "I'm a chef and I hate...";
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
