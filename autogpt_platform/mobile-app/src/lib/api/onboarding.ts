import { apiRequest } from "./client";

export const onboardingKeys = {
  completed: ["onboarding", "completed"] as const,
};

export interface OnboardingProfile {
  user_name: string;
  user_role: string;
  pain_points: string[];
}

export type BrainDumpBody =
  | { recording_id: string; input_mode: "typed"; text: string }
  | { recording_id: string; input_mode: "skipped" };

export interface OnboardingPlan {
  ready: boolean;
  team: { name: string }[];
}

export interface OnboardingHire {
  failed: { name: string }[];
}

export function getOnboardingCompleted(signal?: AbortSignal) {
  return apiRequest<{ is_completed: boolean }>("/api/onboarding/completed", {
    signal,
  });
}

export function submitOnboardingProfile(
  body: OnboardingProfile,
  signal?: AbortSignal,
) {
  return apiRequest<unknown>("/api/onboarding/profile", {
    method: "POST",
    body,
    signal,
  });
}

export function finalizeBrainDump(body: BrainDumpBody, signal?: AbortSignal) {
  return apiRequest<unknown>("/api/onboarding/brain-dump/finalize", {
    method: "POST",
    body,
    signal,
  });
}

export function getOnboardingPlan(signal?: AbortSignal) {
  return apiRequest<OnboardingPlan>("/api/onboarding/brain-dump/plan", { signal });
}

export function hireOnboardingPlan(signal?: AbortSignal) {
  return apiRequest<OnboardingHire>("/api/onboarding/brain-dump/plan/hire", {
    method: "POST",
    signal,
  });
}

export function completeOnboarding(signal?: AbortSignal) {
  return apiRequest<unknown>("/api/onboarding/step", {
    method: "POST",
    query: { step: "ONBOARDING_COMPLETE" },
    signal,
  });
}
