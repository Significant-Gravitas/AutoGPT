import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";

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
  return session.background_outcome ?? null;
}
