import { emailSummaryScript } from "./emailSummaryScript";
import { monitorPricingScript } from "./monitorPricingScript";
import type { TourChat } from "./types";

export const TOUR_SESSION_ID = "tour-demo-session";
export const EMAIL_SESSION_ID = "tour-demo-2";

export const tourChats: TourChat[] = [
  {
    id: TOUR_SESSION_ID,
    title: "Watch competitor pricing",
    updatedAt: "2026-06-30T10:05:00Z",
    script: monitorPricingScript,
  },
  {
    id: EMAIL_SESSION_ID,
    title: "Summarize my weekly emails",
    updatedAt: "2026-06-29T09:10:00Z",
    script: emailSummaryScript,
  },
];

export function getTourChat(id: string): TourChat {
  return tourChats.find((chat) => chat.id === id) ?? tourChats[0];
}

export const tourSessionIds = tourChats.map((chat) => chat.id);
