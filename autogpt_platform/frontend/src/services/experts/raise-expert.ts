import {
  createRaisedExpert,
  generateExpertAvatar,
  getExpertAvatarGeneration,
  getGetExpertQueryKey,
  getListExpertsQueryKey,
  updateExpertAvatar,
} from "@/app/api/__generated__/endpoints/experts/experts";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertAvatarRequestCategory } from "@/app/api/__generated__/models/expertAvatarRequestCategory";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import { uploadSubmissionMediaDirect } from "@/lib/direct-upload";
import type { QueryClient } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
import { invalidateExpertRosterQueries } from "./invalidate-experts";

export interface RaiseInput {
  id: string;
  name: string;
  jobTitle: string;
  role: string;
  category: ExpertAvatarRequestCategory;
  color: string;
  imageFile: File | null;
  /** A local preview of `imageFile`, shown until the upload lands. */
  imagePreviewUrl: string | null;
}

type RaiseStatus = "pending" | "created" | "failed";

const statuses = new Map<string, RaiseStatus>();
const creations = new Map<string, Promise<Expert>>();
const listeners = new Set<() => void>();

// The platform default the server reports for a raised expert with no budget
// of its own (`expert_weekly_credit_budget_default`, $5). Showing it at once
// keeps the budget meter from reading "No budget" until the create lands.
const DEFAULT_WEEKLY_BUDGET_CREDITS = 500;
const AVATAR_POLL_MS = 3_000;
const AVATAR_POLL_LIMIT = 80;

function setStatus(id: string, status: RaiseStatus) {
  statuses.set(id, status);
  listeners.forEach((listener) => listener());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

/** Where a raise started in this tab stands: `undefined` for any expert that
 *  was not raised here, which is every expert after a reload. */
export function useRaiseStatus(expertId: string) {
  return useSyncExternalStore(
    subscribe,
    () => statuses.get(expertId),
    () => undefined,
  );
}

/** Resolves once the server row for a raise started in this tab exists. */
export function waitForRaise(expertId: string): Promise<unknown> {
  return creations.get(expertId) ?? Promise.resolve();
}

/** Optimistic raise: the expert is in the cache before this returns, so the
 *  caller can open its page at once. The server create, the avatar upload
 *  and the avatar generation all run behind it. */
export function startRaise(queryClient: QueryClient, input: RaiseInput) {
  const optimistic = buildOptimisticExpert(input);
  queryClient.setQueryData(getGetExpertQueryKey(input.id), {
    status: 200,
    data: optimistic,
    headers: new Headers(),
  });
  setStatus(input.id, "pending");

  const creation = createOnServer(input).then((expert) => {
    queryClient.setQueryData(getGetExpertQueryKey(input.id), {
      status: 200,
      data: {
        ...expert,
        avatar_url: input.imagePreviewUrl ?? expert.avatar_url,
      },
      headers: new Headers(),
    });
    setStatus(input.id, "created");
    void invalidateExpertRosterQueries(queryClient);
    return expert;
  });
  creations.set(input.id, creation);
  creation.catch(() => {
    setStatus(input.id, "failed");
    queryClient.removeQueries({ queryKey: getGetExpertQueryKey(input.id) });
  });

  void creation
    .then(() => attachAvatar(queryClient, input))
    .catch(() => {
      // The expert keeps its default avatar; it can be changed from its page.
    });
  return creation;
}

async function createOnServer(input: RaiseInput): Promise<Expert> {
  const response = await createRaisedExpert({
    id: input.id,
    name: input.name,
    role: input.role,
    job_title: input.jobTitle || null,
    color: input.color,
  });
  return (response.data as RaiseResult).expert;
}

async function attachAvatar(queryClient: QueryClient, input: RaiseInput) {
  const avatarUrl = input.imageFile
    ? await uploadSubmissionMediaDirect(input.imageFile, "expert-avatar")
    : await generateAvatar(input.category);
  if (!avatarUrl) return;
  await updateExpertAvatar(input.id, { avatar_url: avatarUrl });
  await Promise.all([
    queryClient.invalidateQueries({
      queryKey: getGetExpertQueryKey(input.id),
    }),
    queryClient.invalidateQueries({ queryKey: getListExpertsQueryKey() }),
  ]);
}

async function generateAvatar(
  category: ExpertAvatarRequestCategory,
): Promise<string | null> {
  const started = await generateExpertAvatar({ category });
  const jobId = started.status === 202 ? started.data.id : undefined;
  if (!jobId) return null;
  for (let attempt = 0; attempt < AVATAR_POLL_LIMIT; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, AVATAR_POLL_MS));
    const job = await getExpertAvatarGeneration(jobId);
    if (job.status !== 200) return null;
    if (job.data.status === "complete") return job.data.avatar_url ?? null;
    if (job.data.status === "failed") return null;
  }
  return null;
}

function buildOptimisticExpert(input: RaiseInput): Expert {
  return {
    id: input.id,
    name: input.name,
    avatar_url: input.imagePreviewUrl,
    color: input.color,
    role: input.role,
    job_title: input.jobTitle || null,
    tagline: null,
    bio: null,
    skills: [],
    categories: [input.category],
    identity: "",
    voice_preferences: "",
    boundaries: "",
    protected_soul_rules: [],
    is_template: false,
    source_template_id: null,
    is_archived: false,
    workflows: [],
    weekly_budget: DEFAULT_WEEKLY_BUDGET_CREDITS,
    weekly_spend: 0,
  };
}
