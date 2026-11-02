import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import type { Step, StepLayout } from "./store";
import { z } from "zod";
import { normalizeOnboardingProfile } from "./helpers";

export type WizardStepKey = keyof StepLayout;
const STEP_KEYS: WizardStepKey[] = [
  "team",
  "autopilot",
  "role",
  "painPoints",
  "hire",
  "subscription",
  "connect",
  "preparing",
];

export function stepKey(steps: StepLayout, step: number): WizardStepKey {
  return STEP_KEYS.find((key) => steps[key] === step) ?? "role";
}

export function orderedSteps(steps: StepLayout) {
  return STEP_KEYS.filter((key) => steps[key] !== undefined).sort(
    (a, b) => steps[a]! - steps[b]!,
  );
}

const stepSchema = z.enum([
  "team",
  "autopilot",
  "role",
  "painPoints",
  "hire",
  "subscription",
  "connect",
  "preparing",
]);
function isAllowedCharacter(character: string) {
  const code = character.charCodeAt(0);
  return (
    (code >= 32 && code !== 127) || code === 9 || code === 10 || code === 13
  );
}

export function cleanWizardText(value: string, limit: number) {
  return Array.from(value).filter(isAllowedCharacter).join("").slice(0, limit);
}

function wizardText(max: number, min = 0) {
  return z
    .string()
    .min(min)
    .max(max)
    .refine((value) => Array.from(value).every(isAllowedCharacter));
}

const progressSchema = z
  .object({
    version: z.literal(1),
    currentStep: stepSchema,
    completedSteps: z.array(stepSchema).max(8),
    role: wizardText(100),
    otherRole: wizardText(100),
    painPoints: z.array(wizardText(200, 1)).max(20),
    otherPainPoint: wizardText(2000),
    selectedBilling: z.enum(["monthly", "yearly"]),
    hasUserSelectedBilling: z.boolean(),
    selectedCountryCode: z.string().regex(/^[A-Z]{2}$/),
    hiredTemplateIds: z.array(wizardText(128, 1)).max(100),
  })
  .strict();

export function readProgress(value: unknown): OnboardingWizardProgress | null {
  const result = progressSchema.safeParse(value);
  return result.success ? result.data : null;
}

export function resumeStep({
  progress,
  steps,
  requestedStep,
}: {
  progress: OnboardingWizardProgress | null;
  steps: StepLayout;
  requestedStep?: string | null;
}): Step {
  const order = orderedSteps(steps);
  const completed = new Set(progress?.completedSteps ?? []);
  // Payment must be confirmed by the subscription query, never by draft data
  // or a success query parameter. A confirmed plan removes this layout slot.
  completed.delete("subscription");
  if (progress && !normalizeOnboardingProfile(progress).role.trim())
    completed.delete("role");
  const ceilingKey = order.find((key) => !completed.has(key)) ?? "preparing";
  const ceiling = steps[ceilingKey]!;
  const requested = order.find((key) => key === requestedStep);
  const saved = progress?.currentStep;
  const target =
    requested ?? (saved && steps[saved] !== undefined ? saved : ceilingKey);
  return Math.min(steps[target]!, ceiling) as Step;
}

export function progressStorageKey(userID: string) {
  return `autogpt:onboarding-progress:${userID}`;
}

export function readLocalProgress(userID: string) {
  try {
    const value = JSON.parse(
      localStorage.getItem(progressStorageKey(userID)) ?? "null",
    );
    const progress = readProgress(value?.progress);
    return progress && Number.isInteger(value.revision) && value.revision >= 0
      ? {
          progress,
          pending: value.pending === true,
          revision: value.revision as number,
        }
      : null;
  } catch {
    return null;
  }
}

export function writeLocalProgress(
  userID: string,
  progress: OnboardingWizardProgress,
  pending: boolean,
  revision: number,
) {
  try {
    localStorage.setItem(
      progressStorageKey(userID),
      JSON.stringify({ progress, pending, revision }),
    );
    return true;
  } catch {
    return false;
  }
}

export function clearLocalProgress(userID: string) {
  try {
    localStorage.removeItem(progressStorageKey(userID));
  } catch {
    // A confirmed server completion still prevents re-entering the wizard.
  }
}

export function sameProgress(
  left: OnboardingWizardProgress,
  right: OnboardingWizardProgress | null,
) {
  return (
    JSON.stringify(readProgress(left)) === JSON.stringify(readProgress(right))
  );
}
