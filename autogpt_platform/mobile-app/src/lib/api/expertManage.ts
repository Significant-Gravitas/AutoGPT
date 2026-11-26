import { apiRequest } from "./client";
import type { ExpertDetail } from "./expertActivity";

/** `AutopilotMode` in backend/copilot/autopilot_mode.py: how often an expert asks first. */
export type AutopilotMode = "ask_first" | "auto" | "unsupervised";

/** `ExpertSoulUpdate`: who the expert is and how it works. */
export interface ExpertSoul {
  name: string;
  identity: string;
  voice_preferences: string;
  boundaries: string;
}

function expertPath(expertID: string, route: string) {
  return `/api/experts/${encodeURIComponent(expertID)}/${route}`;
}

export function updateExpertSoul(expertID: string, soul: ExpertSoul) {
  return apiRequest<ExpertDetail>(expertPath(expertID, "soul"), {
    method: "PATCH",
    body: soul,
  });
}

/** In credits, 100 to the dollar; null falls back to the default, 0 lifts the cap. */
export function updateExpertBudget(expertID: string, weeklyBudget: number | null) {
  return apiRequest<ExpertDetail>(expertPath(expertID, "budget"), {
    method: "PATCH",
    body: { weekly_budget: weeklyBudget },
  });
}

/** Null clears it back to the platform default. A 403 means modes are off for this account. */
export function updateExpertMode(expertID: string, mode: AutopilotMode | null) {
  return apiRequest<ExpertDetail>(expertPath(expertID, "mode"), {
    method: "PATCH",
    body: { autopilot_mode: mode },
  });
}
