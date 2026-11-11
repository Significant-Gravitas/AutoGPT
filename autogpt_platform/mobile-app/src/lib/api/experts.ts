import { apiRequest } from "./client";

export interface Expert {
  id: string;
  name: string;
  avatar_url: string | null;
  role: string;
  job_title: string | null;
  tagline: string | null;
  categories: string[];
  last_run_at: string | null;
  last_run_status: string | null;
  schedules_paused_at: string | null;
  setup_status: string;
}

export const expertKeys = {
  list: ["experts"] as const,
};

export function listExperts() {
  return apiRequest<Expert[]>("/api/experts");
}
