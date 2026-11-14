import { clearAccessToken } from "@/lib/auth/token";

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
}

export const expertKeys = {
  list: ["experts"] as const,
};

export function listExperts() {
  return apiRequest<Expert[]>("/api/experts");
}

export interface CreateExpertBody {
  name: string;
  role: string | null;
  job_title: string | null;
  avatar_url: string | null;
  color: string | null;
}

interface RaiseResult {
  expert: Expert;
}

/** A refused create; a 409 names the limit it hit in `code` and `limit`. */
export class CreateExpertError extends ApiError {
  code: string | null;
  limit: number | null;

  constructor(status: number, message: string, code: string | null, limit: number | null) {
    super(status, message);
    this.name = "CreateExpertError";
    this.code = code;
    this.limit = limit;
  }
}

// Not `apiRequest`: its error keeps only a string `detail`, and the limit
// a 409 hit arrives as `{ code, limit }`.
export async function createExpert(body: CreateExpertBody) {
  let response = await postRaise(body);
  if (response.status === 401) {
    clearAccessToken();
    response = await postRaise(body);
  }
  if (!response.ok) throw await createExpertError(response);
  return (await response.json()) as RaiseResult;
}

async function postRaise(body: CreateExpertBody) {
  return fetch(apiUrl("/api/experts/raise"), {
    method: "POST",
    headers: { ...(await authHeaders()), "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

async function createExpertError(response: Response) {
  let detail: unknown = null;
  try {
    detail = ((await response.json()) as { detail?: unknown }).detail ?? null;
  } catch {
    // Not JSON: only the status is known.
  }
  const fields = typeof detail === "object" && detail !== null
    ? (detail as { code?: unknown; limit?: unknown })
    : {};
  return new CreateExpertError(
    response.status,
    typeof detail === "string" ? detail : `Request failed (${response.status})`,
    typeof fields.code === "string" ? fields.code : null,
    typeof fields.limit === "number" ? fields.limit : null,
  );
}
