import { asRecord, asString, nonEmptyString } from "../lib/json";
import type { BackendClient } from "./backend";

/** The lightweight identity the chat surfaces need, including let-go experts. */
export interface ExpertIdentity {
  id: string;
  name: string;
  avatarURL: string | null;
  role: string | null;
  jobTitle: string | null;
  isArchived: boolean;
}

export interface ExpertProfile extends ExpertIdentity {
  tagline: string | null;
  lastRunAt: string | null;
  lastRunStatus: string | null;
  setupStatus: string | null;
  schedulesPausedAt: string | null;
}

export async function listExpertIdentities(
  api: BackendClient,
  signal?: AbortSignal,
): Promise<ExpertIdentity[]> {
  const body = await api.json<unknown>("/api/experts/identities", { signal });
  return (Array.isArray(body) ? body : [])
    .map(toIdentity)
    .filter((identity): identity is ExpertIdentity => identity !== null);
}

/** The active hired roster with status fields for the Company screen. */
export async function listExperts(
  api: BackendClient,
  signal?: AbortSignal,
): Promise<ExpertProfile[]> {
  const body = await api.json<unknown>("/api/experts", { signal });
  return (Array.isArray(body) ? body : [])
    .map((raw) => {
      const identity = toIdentity(raw);
      const record = asRecord(raw);
      if (!identity || !record) return null;
      return {
        ...identity,
        tagline: nonEmptyString(record.tagline),
        lastRunAt: asString(record.last_run_at),
        lastRunStatus: nonEmptyString(record.last_run_status),
        setupStatus: nonEmptyString(record.setup_status),
        schedulesPausedAt: asString(record.schedules_paused_at),
      };
    })
    .filter(
      (profile): profile is ExpertProfile =>
        profile !== null && !profile.isArchived,
    );
}

function toIdentity(raw: unknown): ExpertIdentity | null {
  const record = asRecord(raw);
  const id = nonEmptyString(record?.id);
  const name = nonEmptyString(record?.name);
  if (!record || !id || !name) return null;
  return {
    id,
    name,
    avatarURL: nonEmptyString(record.avatar_url),
    role: nonEmptyString(record.role),
    jobTitle: nonEmptyString(record.job_title),
    isArchived: record.is_archived === true,
  };
}
