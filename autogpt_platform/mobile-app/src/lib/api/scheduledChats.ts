import { apiRequest } from "./client";

/** `ScheduledChatCadence` in backend/copilot/scheduled_chats/cadence_service.py. */
export interface ScheduledChatCadence {
  schedule_key: string;
  title: string;
  /** How often it runs now, in words, when that can be said. */
  current: string | null;
  timezone: string;
  /** Least drastic first; empty when it already runs once a week or less. */
  options: { cron: string; label: string }[];
}

export const scheduledChatKeys = {
  cadence: (scheduleKey: string) => ["schedules", "followups", scheduleKey, "cadence"] as const,
};

function followupPath(scheduleKey: string, route: string) {
  return `/api/schedules/followups/${encodeURIComponent(scheduleKey)}/${route}`;
}

/** A 404 means it was not paused; a 503 is safe to try again. */
export function resumeScheduledChat(scheduleKey: string) {
  return apiRequest<{ resumed: boolean }>(followupPath(scheduleKey, "resume"), {
    method: "POST",
  });
}

/** A 404 means no cadence to change; a 409 that it is paused. */
export function getCadence(scheduleKey: string) {
  return apiRequest<ScheduledChatCadence>(followupPath(scheduleKey, "cadence"));
}

/** `cron` must be one of the offered options; anything else is a 400. */
export function setCadence(scheduleKey: string, cron: string) {
  return apiRequest<ScheduledChatCadence>(followupPath(scheduleKey, "cadence"), {
    method: "POST",
    body: { cron },
  });
}
