import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { asObject } from "./resultHelpers";

export function isSessionPending(
  session: SessionDetailResponse,
  background: boolean,
) {
  return (
    isSessionLive(session) || (background && !getBackgroundOutcome(session))
  );
}

export function isSessionLive(session: SessionDetailResponse): boolean {
  if (session.active_stream) return true;
  const status = session.chat_status?.toLowerCase();
  return status === "running" || status === "queued";
}

export function getBackgroundOutcome(session: SessionDetailResponse) {
  for (const raw of session.messages) {
    const metadata = asObject(asObject(raw)?.metadata);
    if (typeof metadata?.background_task_outcome === "string") {
      return metadata.background_task_outcome;
    }
  }
  return null;
}
