import type { Expert } from "@/app/api/__generated__/models/expert";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { Task } from "@/app/api/__generated__/models/task";
import type { TaskStatus } from "@/app/api/__generated__/models/taskStatus";

export function makeTask(
  id: string,
  title: string,
  status: TaskStatus,
  ownerExpertID: string | null = null,
): Task {
  return {
    id,
    user_id: "user-1",
    title,
    description: null,
    status,
    owner_expert_id: ownerExpertID,
    created_by: "USER",
    session_ids: [],
    created_at: new Date("2026-11-09T10:00:00Z"),
    updated_at: new Date("2026-11-10T09:00:00Z"),
  };
}

export function makeSession(
  id: string,
  title: string,
  expertID: string | null,
  extra: Partial<SessionSummaryResponse> = {},
): SessionSummaryResponse {
  return {
    id,
    title,
    created_at: "2026-11-10T09:00:00Z",
    updated_at: "2026-11-10T09:00:00Z",
    chat_status: "idle",
    is_processing: false,
    expert_id: expertID,
    ...extra,
  };
}

export function makeExpert(id: string, name: string): Expert {
  return {
    id,
    name,
    avatar_url: null,
    role: "Role",
    tagline: null,
    bio: null,
    skills: [],
    identity: "identity",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    pod_id: null,
  };
}
