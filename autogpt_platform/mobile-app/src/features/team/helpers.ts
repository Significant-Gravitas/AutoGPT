import type { Expert } from "@/lib/api/experts";
import { FRONTEND_URL } from "@/lib/env";

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

/**
 * The avatar URL comes from user data, so only web images load: `https:`
 * anywhere, `http:` only when the frontend itself runs on it (local dev).
 */
export function avatarSource(expert: Pick<Expert, "avatar_url">) {
  const url = expert.avatar_url?.trim();
  if (!url) return null;
  if (url.startsWith("/") && !url.startsWith("//")) return FRONTEND_URL + url;
  if (/^https:\/\//i.test(url)) return url;
  if (/^http:\/\//i.test(url) && /^http:\/\//i.test(FRONTEND_URL)) return url;
  return null;
}

export function relativeTime(iso: string, now = Date.now()) {
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return null;
  const elapsed = Math.max(0, now - then);
  if (elapsed < MINUTE) return "just now";
  if (elapsed < HOUR) return `${Math.floor(elapsed / MINUTE)} min ago`;
  if (elapsed < DAY) return `${Math.floor(elapsed / HOUR)} h ago`;
  return `${Math.floor(elapsed / DAY)} d ago`;
}

export function statusLine(
  expert: Pick<Expert, "setup_status" | "schedules_paused_at" | "last_run_at">,
  now = Date.now(),
) {
  if (expert.setup_status !== "ready") return "Getting set up";
  if (expert.schedules_paused_at) return "Paused";
  if (expert.last_run_at) {
    const ago = relativeTime(expert.last_run_at, now);
    return ago ? `Last active ${ago}` : null;
  }
  return null;
}

export function roleLine(expert: Pick<Expert, "job_title" | "role">) {
  return expert.job_title || expert.role || null;
}
