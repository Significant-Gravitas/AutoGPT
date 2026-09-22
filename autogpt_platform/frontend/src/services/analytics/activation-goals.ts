import { analytics } from "@/services/analytics";

/**
 * DataFast goals for the activation actions a person takes in the browser.
 *
 * These exist for marketing attribution (which channel brings users who
 * actually run things), so they fire only on human clicks, never for
 * server-triggered work. The server-side record of every run lives in
 * PostHog (`run_agent`, `run_autopilot`, ...) and the `analytics.*` views;
 * this file is only the DataFast mirror of the two goals it can see.
 *
 * `run_agent` used to fire from a single library modal, so builder runs,
 * re-runs and the builder's run dialog never counted. Every human run path
 * goes through `trackAgentRunGoal` now.
 */

export type AgentRunSurface = "library" | "builder" | "rerun";
export type ScheduleSurface = "library" | "builder";

interface AgentRef {
  id: string;
  name?: string | null;
}

export function trackAgentRunGoal(agent: AgentRef, surface: AgentRunSurface) {
  sendGoal("run_agent", { id: agent.id, name: agent.name ?? "", surface });
}

export function trackScheduleCreatedGoal(
  agent: AgentRef,
  surface: ScheduleSurface,
) {
  sendGoal("schedule_agent", { id: agent.id, name: agent.name ?? "", surface });
}

function sendGoal(name: string, metadata: Record<string, string>) {
  try {
    analytics.sendDatafastEvent(name, metadata);
  } catch {
    // Attribution must never break the action it describes.
  }
}
