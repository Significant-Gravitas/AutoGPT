import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";

export const NOW = new Date("2026-08-09T12:00:00Z");

export const ada = { id: "ada", name: "Ada", role: "Ops", avatar_url: null };
export const leo = { id: "leo", name: "Leo", role: "Sales", avatar_url: null };

interface Options {
  expert?: HomeAttentionItem["expert"];
  session?: string;
  priority?: HomeAttentionItem["priority"];
}

// A held call's row as `_gate_attention` writes it (backend/api/features/home/attention.py).
export function homeHeldItem(
  review: PendingHumanReviewModel,
  { expert = ada, session = "s1", priority = "normal" }: Options = {},
): HomeAttentionItem {
  const headline = (review.payload as { headline: HeadlinePayload }).headline;
  const title = headline.object
    ? `${headline.ask} “${headline.object}”`
    : headline.ask;
  return {
    id: `approval-${review.node_exec_id}`,
    kind: "approval",
    priority,
    title,
    headline: { ask: headline.ask, object: headline.object ?? null },
    description: "Otto is waiting for your approval.",
    why_it_matters: "Nothing runs until you approve it.",
    expert,
    created_at: review.created_at,
    primary_action: {
      label: "Open chat",
      href: `/copilot?sessionId=${session}`,
    },
    review: {
      ...review,
      session_id: session,
      graph_exec_id: `copilot-session-${session}`,
    },
  };
}

interface HeadlinePayload {
  ask: string;
  object?: string | null;
}

export function makeDashboard(
  attention: HomeAttentionItem[],
): HomeDashboardResponse {
  return {
    generated_at: NOW,
    timezone: "UTC",
    attention,
    briefing: {
      generated_at: NOW,
      window_started_at: new Date("2026-08-08T12:00:00Z"),
      completed_count: 0,
      failed_count: 0,
      routine_count: 0,
      outcomes: [],
      author: { kind: "autopilot", name: "Otto", role: "Head of AI" },
    },
    active_tasks: [],
    upcoming_tasks: [],
    team: { total: 0, ready: 0, working: 0, needs_attention: 0 },
    agents: [],
    week: {
      run_count: 0,
      completed_count: 0,
      review_count: 0,
      failed_count: 0,
      total_runtime_seconds: 0,
      timed_run_count: 0,
      total_cost_cents: 0,
      credits_balance: 0,
      daily: [],
    },
  };
}
