import { assertSameSession, clearAccessToken, sessionMark } from "@/lib/auth/token";

import { ApiError, apiRequest, apiUrl, authHeaders } from "./client";

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
  /** The roster template this expert was hired from; null for a created one. */
  source_template_id: string | null;
}

export const expertKeys = {
  list: ["experts"] as const,
};

// Takes React Query's context as is, so a dropped query cancels its fetch.
export function listExperts({ signal }: { signal?: AbortSignal } = {}) {
  return apiRequest<Expert[]>("/api/experts", { signal });
}

export interface BundledSkill {
  id: string;
  slug: string;
  title: string;
  description: string;
}

/** A step the expert takes in its first days (`ExpertDayOneItem`). */
export interface DayOneItem {
  title: string;
  description: string;
  timing: string;
}

/** A workflow that comes with the expert (`ExpertWorkflowRef`). */
export interface ExpertWorkflow {
  id: string;
  name: string | null;
  description: string | null;
  schedule_cron: string | null;
  integration_providers?: string[];
}

/**
 * A roster expert as the marketplace shows it. The profile's fields are
 * optional: a list cached by an older build of the app has none of them.
 */
export interface ExpertTemplate extends Expert {
  bio: string | null;
  bundled_skills: BundledSkill[];
  day_one?: DayOneItem[];
  workflows?: ExpertWorkflow[];
  protected_soul_rules?: string[];
  /** In credits, 100 to the dollar. */
  weekly_budget?: number | null;
}

export const templateKeys = {
  list: ["expert-templates"] as const,
};

// Unpaginated: the roster is small, and the app filters it on the phone.
export function listTemplates() {
  return apiRequest<ExpertTemplate[]>("/api/experts/templates");
}

export interface CreateExpertBody {
  name: string;
  role: string | null;
  job_title: string | null;
  avatar_url: string | null;
  color: string | null;
}

interface ExpertResult {
  expert: Expert;
}

/** A refused create or hire; a 409 names the limit it hit in `code` and `limit`. */
export class ExpertLimitError extends ApiError {
  code: string | null;
  limit: number | null;

  constructor(status: number, message: string, code: string | null, limit: number | null) {
    super(status, message);
    this.name = "ExpertLimitError";
    this.code = code;
    this.limit = limit;
  }
}

export function createExpert(body: CreateExpertBody) {
  return postExpertJSON("/api/experts/raise", body);
}

// No `surface`: the server then sends the new expert's hello itself.
export function hireExpert(templateID: string) {
  return postExpertJSON("/api/experts", { template_id: templateID });
}

// Not `apiRequest`: its error keeps only a string `detail`, and the limit
// a 409 hit arrives as `{ code, limit }`.
async function postExpertJSON(path: string, body: unknown) {
  const mark = sessionMark();
  let response = await post(path, body);
  if (response.status === 401) {
    // Never act for an account that signed in after this started.
    assertSameSession(mark);
    clearAccessToken();
    response = await post(path, body);
    assertSameSession(mark);
  }
  if (!response.ok) {
    const error = await limitError(response);
    assertSameSession(mark);
    throw error;
  }
  return (await response.json()) as ExpertResult;
}

async function post(path: string, body: unknown) {
  return fetch(apiUrl(path), {
    method: "POST",
    headers: { ...(await authHeaders()), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function limitError(response: Response) {
  let detail: unknown = null;
  try {
    detail = ((await response.json()) as { detail?: unknown }).detail ?? null;
  } catch {
    // Not JSON: only the status is known.
  }
  const fields = typeof detail === "object" && detail !== null
    ? (detail as { code?: unknown; limit?: unknown })
    : {};
  return new ExpertLimitError(
    response.status,
    typeof detail === "string" ? detail : `Request failed (${response.status})`,
    typeof fields.code === "string" ? fields.code : null,
    typeof fields.limit === "number" ? fields.limit : null,
  );
}
