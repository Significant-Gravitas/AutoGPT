import { apiRequest } from "./client";

// Only the fields the app reads; see backend/api/features/home/models.py.
export interface HomeExpert {
  id: string;
  name: string;
  role: string;
  avatar_url: string | null;
}

export interface HomeAttentionItem {
  id: string;
  kind: "approval" | "setup" | "paused" | "credits" | "question";
  priority: "high" | "normal";
  title: string;
  description: string;
  expert?: HomeExpert | null;
  primary_action: { label: string; href: string };
}

export interface HomeActiveTask {
  id: string;
  title: string;
  status: "running" | "queued";
  expert?: HomeExpert | null;
}

export interface HomeUpcomingTask {
  id: string;
  title: string;
  expert?: HomeExpert | null;
  next_run_time: string;
}

export interface HomeDashboard {
  attention: HomeAttentionItem[];
  briefing: { completed_count: number };
  active_tasks: HomeActiveTask[];
  upcoming_tasks: HomeUpcomingTask[];
}

export const homeKeys = {
  dashboard: ["home"] as const,
};

export function getHome() {
  return apiRequest<HomeDashboard>("/api/home");
}
