import { apiRequest } from "./client";

/** `MemoryScopeOverview` in backend/api/features/memory/models.py: the account's counts. */
export interface MemoryOverview {
  facts: number;
  entities: number;
  episodes: number;
}

/** One live memory, newest first in the list. */
export interface MemoryFact {
  uuid: string;
  fact: string | null;
  name: string | null;
  source: string;
  target: string;
  created_at: string | null;
}

export interface MemoryFactList {
  items: MemoryFact[];
}

/** What Otto keeps about the person's work; `updated_at` is null when nothing is kept. */
export interface MemoryUnderstanding {
  updated_at: string | null;
  user_name: string | null;
  job_title: string | null;
  business_name: string | null;
  industry: string | null;
  business_size: string | null;
  user_role: string | null;
  key_workflows: string[];
  daily_activities: string[];
  pain_points: string[];
  bottlenecks: string[];
  manual_tasks: string[];
  automation_goals: string[];
  current_software: string[];
  existing_automation: string[];
  additional_notes: string | null;
}

export const memoryKeys = {
  all: ["memory"] as const,
  overview: ["memory", "overview"] as const,
  facts: ["memory", "facts"] as const,
  understanding: ["memory", "understanding"] as const,
};

export function getMemoryOverview() {
  return apiRequest<MemoryOverview>("/api/memory/overview");
}

export function listMemoryFacts(limit = 50) {
  return apiRequest<MemoryFactList>("/api/memory/facts", { query: { limit } });
}

/** A 409 means memory is busy and a 500 that clean-up didn't finish; both can be retried. */
export function forgetMemoryFact(uuid: string) {
  return apiRequest<{ forgotten: boolean }>(
    `/api/memory/facts/${encodeURIComponent(uuid)}`,
    { method: "DELETE" },
  );
}

/** Wipes every memory and what Otto knows about the person's work, for good. */
export function eraseMemory() {
  return apiRequest<{ erased: boolean }>("/api/memory", { method: "DELETE" });
}

export function getUnderstanding() {
  return apiRequest<MemoryUnderstanding>("/api/memory/understanding");
}

export function forgetUnderstanding() {
  return apiRequest<{ forgotten: boolean }>("/api/memory/understanding", {
    method: "DELETE",
  });
}
