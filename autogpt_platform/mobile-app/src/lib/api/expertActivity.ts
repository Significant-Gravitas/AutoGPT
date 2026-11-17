import { apiRequest } from "./client";

/** `ExpertRunStatus` in backend/api/features/experts/models.py. */
export type ExpertRunStatus =
  | "incomplete"
  | "queued"
  | "running"
  | "completed"
  | "terminated"
  | "failed"
  | "review";

/** The fields of `ExpertRun` the panel reads. */
export interface ExpertRun {
  execution_id: string;
  agent_name: string;
  status: ExpertRunStatus;
  started_at: string | null;
  ended_at: string | null;
}

/** `SandboxSummary` in backend/copilot/computer.py. */
export interface SandboxSummary {
  sandbox_id: string;
  state: "running" | "paused";
  started_at: string;
}

/** The fields of `ComputerInfo` the panel reads. */
export interface ComputerInfo {
  e2b_active: boolean;
  box: SandboxSummary | null;
  screen_on: boolean;
}

/**
 * The fields both `GraphExecutionJobInfo` and `CopilotTurnJobInfo` share.
 * `next_run_time` is "" for a paused job or a one-off that already fired.
 */
export interface ScheduleInfo {
  id: string;
  name: string;
  next_run_time: string;
  expert_id?: string | null;
}

/** The fields of the full `Expert` from `GET /experts/{id}` the panel reads. */
export interface ExpertDetail {
  id: string;
  schedules_paused_at: string | null;
  workflows: { schedule_id: string | null }[];
}

export const expertActivityKeys = {
  detail: (expertID: string) => ["experts", expertID, "detail"] as const,
  runs: (expertID: string) => ["experts", expertID, "runs"] as const,
  computer: (expertID: string) => ["experts", expertID, "computer"] as const,
  graphSchedules: ["schedules", "graphs"] as const,
  followupSchedules: ["schedules", "followups"] as const,
};

export function getExpertDetail(expertID: string) {
  return apiRequest<ExpertDetail>(`/api/experts/${encodeURIComponent(expertID)}`);
}

export function listExpertRuns(expertID: string) {
  return apiRequest<ExpertRun[]>(
    `/api/experts/${encodeURIComponent(expertID)}/runs`,
  );
}

/** Read-only: lists the box without waking it. */
export function getExpertComputer(expertID: string) {
  return apiRequest<ComputerInfo>(
    `/api/experts/${encodeURIComponent(expertID)}/computer`,
  );
}

/** Workflow schedules, paused ones included. */
export function listGraphSchedules() {
  return apiRequest<ScheduleInfo[]>("/api/schedules");
}

/** Scheduled chats: an expert's routines fire as these. */
export function listFollowupSchedules() {
  return apiRequest<ScheduleInfo[]>("/api/schedules/followups");
}

/** Undoes the expert-wide pause; there is no route to pause one routine. */
export function resumeExpertSchedules(expertID: string) {
  return apiRequest<ExpertDetail>(
    `/api/experts/${encodeURIComponent(expertID)}/schedules/resume`,
    { method: "POST" },
  );
}
