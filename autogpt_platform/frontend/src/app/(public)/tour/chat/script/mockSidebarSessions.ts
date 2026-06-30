import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";

export const TOUR_SESSION_ID = "tour-demo-session";

export const mockSidebarSessions: SessionSummaryResponse[] = [
  {
    id: TOUR_SESSION_ID,
    title: "Watch competitor pricing",
    created_at: "2026-06-30T10:00:00Z",
    updated_at: "2026-06-30T10:05:00Z",
    chat_status: "idle",
    is_processing: false,
  },
  {
    id: "tour-demo-2",
    title: "Summarize my weekly emails",
    created_at: "2026-06-29T09:00:00Z",
    updated_at: "2026-06-29T09:10:00Z",
    chat_status: "idle",
    is_processing: false,
  },
];
