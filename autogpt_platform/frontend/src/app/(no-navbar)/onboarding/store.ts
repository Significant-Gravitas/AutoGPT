import { create } from "zustand";
import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import { cleanWizardText, stepKey, type WizardStepKey } from "./progress";

export const MAX_PAIN_POINT_SELECTIONS = 3;
export type Step = 1 | 2 | 3 | 4 | 5 | 6 | 7;
export const MAX_STEP: Step = 7;

// Every step the wizard can show, numbered for one deployment. Optional
// steps are absent rather than zero so `currentStep === steps.team` can
// never match by accident.
export interface StepLayout {
  team?: number;
  autopilot?: number;
  role: number;
  painPoints: number;
  hire?: number;
  subscription?: number;
  connect?: number;
  preparing: number;
}

export function buildStepLayout({
  hasPaywall = false,
  hasConnect = false,
}: {
  hasPaywall?: boolean;
  hasConnect?: boolean;
}): StepLayout {
  const order: (keyof StepLayout)[] = [
    "team",
    "autopilot",
    "role",
    "painPoints",
    "hire",
    ...(hasPaywall ? (["subscription"] as const) : []),
    ...(!hasPaywall && hasConnect ? (["connect"] as const) : []),
    "preparing",
  ];
  const layout: Partial<Record<keyof StepLayout, number>> = {};
  order.forEach((key, index) => {
    layout[key] = index + 1;
  });
  return layout as StepLayout;
}

export const PAYWALL_LAST_STEPS = buildStepLayout({ hasPaywall: true });
export const NO_PAYWALL_STEPS = buildStepLayout({});
export const SELF_HOST_STEPS = buildStepLayout({ hasConnect: true });

interface OnboardingWizardState {
  currentStep: Step;
  userID: string | null;
  completedSteps: WizardStepKey[];
  flushProgress: (() => Promise<void>) | null;
  // The numbering in force for this session, set by the page hook once the
  // flags resolve; steps that need to name another step (the paywall's
  // Stripe return URLs) read it from here.
  steps: StepLayout;
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
  selectedPlan: string | null;
  selectedBilling: "monthly" | "yearly";
  hasUserSelectedBilling: boolean;
  selectedCountryCode: string;
  /** Templates hired from the recommendation step, so a Back/forward trip
   * through the wizard keeps showing them as hired. */
  hiredTemplateIds: string[];
  /** True while the current step is mid-flight (e.g. the brain dump is
   * being processed) — navigation away must be blocked. Transient, never
   * persisted. */
  isStepBusy: boolean;
  setStepBusy(busy: boolean): void;
  setSteps(steps: StepLayout): void;
  setRole(role: string): void;
  setOtherRole(otherRole: string): void;
  togglePainPoint(painPoint: string): void;
  setOtherPainPoint(otherPainPoint: string): void;
  setSelectedPlan(plan: string | null): void;
  setSelectedBilling(billing: "monthly" | "yearly"): void;
  applyPricingExperimentBilling(billing: "monthly" | "yearly"): void;
  setSelectedCountryCode(code: string): void;
  markHired(templateId: string): void;
  nextStep(): void;
  prevStep(): void;
  goToStep(step: Step): void;
  reset(): void;
}

export const useOnboardingWizardStore = create<OnboardingWizardState>()(
  (set) => ({
    currentStep: 1,
    userID: null,
    completedSteps: [],
    flushProgress: null,
    steps: PAYWALL_LAST_STEPS,
    role: "",
    otherRole: "",
    painPoints: [],
    otherPainPoint: "",
    selectedPlan: null,
    selectedBilling: "monthly",
    hasUserSelectedBilling: false,
    selectedCountryCode: "US",
    hiredTemplateIds: [],
    isStepBusy: false,
    setStepBusy(busy) {
      set({ isStepBusy: busy });
    },
    setSteps(steps) {
      set({ steps });
    },
    setRole(role) {
      set({ role: cleanWizardText(role, 100) });
    },
    setOtherRole(otherRole) {
      set({ otherRole: cleanWizardText(otherRole, 100) });
    },
    togglePainPoint(painPoint) {
      set((state) => {
        const exists = state.painPoints.includes(painPoint);
        if (!exists && state.painPoints.length >= MAX_PAIN_POINT_SELECTIONS)
          return state;
        return {
          painPoints: exists
            ? state.painPoints.filter((p) => p !== painPoint)
            : [...state.painPoints, painPoint],
        };
      });
    },
    setOtherPainPoint(otherPainPoint) {
      set({ otherPainPoint: cleanWizardText(otherPainPoint, 2000) });
    },
    setSelectedPlan(plan) {
      set({ selectedPlan: plan });
    },
    setSelectedBilling(billing) {
      set({ selectedBilling: billing, hasUserSelectedBilling: true });
    },
    applyPricingExperimentBilling(billing) {
      set((state) =>
        state.hasUserSelectedBilling ? state : { selectedBilling: billing },
      );
    },
    setSelectedCountryCode(code) {
      set({ selectedCountryCode: code });
    },
    markHired(templateId) {
      set((state) =>
        state.hiredTemplateIds.includes(templateId)
          ? state
          : { hiredTemplateIds: [...state.hiredTemplateIds, templateId] },
      );
    },
    nextStep() {
      set((state) => ({
        currentStep: Math.min(MAX_STEP, state.currentStep + 1) as Step,
        completedSteps: Array.from(
          new Set([
            ...state.completedSteps,
            stepKey(state.steps, state.currentStep),
          ]),
        ),
      }));
    },
    prevStep() {
      set((state) => ({
        currentStep: Math.max(1, state.currentStep - 1) as Step,
      }));
    },
    goToStep(step) {
      set({ currentStep: step });
    },
    reset() {
      set({
        currentStep: 1,
        userID: null,
        completedSteps: [],
        flushProgress: null,
        isStepBusy: false,
        role: "",
        otherRole: "",
        painPoints: [],
        otherPainPoint: "",
        selectedPlan: null,
        selectedBilling: "monthly",
        hasUserSelectedBilling: false,
        selectedCountryCode: "US",
        hiredTemplateIds: [],
      });
    },
  }),
);

export function snapshotProgress(): OnboardingWizardProgress {
  const state = useOnboardingWizardStore.getState();
  return {
    version: 1,
    currentStep: stepKey(state.steps, state.currentStep),
    completedSteps: state.completedSteps,
    role: state.role,
    otherRole: state.otherRole,
    painPoints: state.painPoints,
    otherPainPoint: state.otherPainPoint,
    selectedBilling: state.selectedBilling,
    hasUserSelectedBilling: state.hasUserSelectedBilling,
    selectedCountryCode: state.selectedCountryCode,
    hiredTemplateIds: state.hiredTemplateIds,
  };
}
