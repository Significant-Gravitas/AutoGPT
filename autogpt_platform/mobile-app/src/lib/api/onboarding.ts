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

export function getOnboardingCompleted() {
  return apiRequest<{ is_completed: boolean }>("/api/onboarding/completed");
}

export function submitOnboardingProfile(body: OnboardingProfile) {
  return apiRequest<unknown>("/api/onboarding/profile", { method: "POST", body });
}

export function finalizeBrainDump(body: BrainDumpBody) {
  return apiRequest<unknown>("/api/onboarding/brain-dump/finalize", {
    method: "POST",
    body,
  });
}

export function getOnboardingPlan() {
  return apiRequest<OnboardingPlan>("/api/onboarding/brain-dump/plan");
}

export function hireOnboardingPlan() {
  return apiRequest<OnboardingHire>("/api/onboarding/brain-dump/plan/hire", {
    method: "POST",
  });
}

export function completeOnboarding() {
  return apiRequest<unknown>("/api/onboarding/step", {
    method: "POST",
    query: { step: "ONBOARDING_COMPLETE" },
  });
}
