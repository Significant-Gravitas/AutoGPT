import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

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

// Builds the layout for a deployment: the paywall comes FIRST on cloud so
// nobody without a plan gets into the wizard, the two intro steps (team, meet
// AutoPilot) follow when the expert team is on, then the profile steps, then
// AutoPilot's hire recommendations right after the brain dump they are read
// from. Self-host has no paywall and instead closes with the "connect a plan
// you already pay for" step, right before Preparing. Numbering is derived, so
// nothing can drift.
export function buildStepLayout({
  hasIntro = false,
  hasHire = false,
  hasPaywall = false,
  hasConnect = false,
}: {
  hasIntro?: boolean;
  hasHire?: boolean;
  hasPaywall?: boolean;
  hasConnect?: boolean;
}): StepLayout {
  const order: (keyof StepLayout)[] = [
    ...(hasPaywall ? (["subscription"] as const) : []),
    ...(hasIntro ? (["team", "autopilot"] as const) : []),
    "role",
    "painPoints",
    ...(hasHire ? (["hire"] as const) : []),
    ...(!hasPaywall && hasConnect ? (["connect"] as const) : []),
    "preparing",
  ];
  const layout: Partial<Record<keyof StepLayout, number>> = {};
  order.forEach((key, index) => {
    layout[key] = index + 1;
  });
  return layout as StepLayout;
}

// The three layouts without the intro steps, spelled out for tests and for
// the store's default.
export const PAYWALL_FIRST_STEPS = {
  subscription: 1,
  role: 2,
  painPoints: 3,
  preparing: 4,
} as const;

export const NO_PAYWALL_STEPS = {
  role: 1,
  painPoints: 2,
  preparing: 3,
} as const;

// Self-host has no paywall; it asks for a model at the end instead — link
// the ChatGPT plan you already pay for — once the profile is in.
export const SELF_HOST_STEPS = {
  role: 1,
  painPoints: 2,
  connect: 3,
  preparing: 4,
} as const;

interface OnboardingWizardState {
  currentStep: Step;
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
  setSelectedPlan(plan: string): void;
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
  persist(
    (set) => ({
      currentStep: 1,
      steps: PAYWALL_FIRST_STEPS,
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
        set({ role });
      },
      setOtherRole(otherRole) {
        set({ otherRole });
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
        set({ otherPainPoint });
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
    {
      name: "onboarding-wizard",
      version: 1,
      // sessionStorage (not localStorage) so abandoning the wizard and
      // closing the tab gives a clean slate next time, matching the
      // STEP_STORAGE_KEY ceiling in useOnboardingPage. SSR-safe: a no-op
      // stub runs during Next.js SSR / vitest where window is undefined —
      // returning undefined would make zustand throw on first getItem.
      storage: createJSONStorage(() =>
        typeof window !== "undefined" && window.sessionStorage
          ? window.sessionStorage
          : { getItem: () => null, setItem: () => {}, removeItem: () => {} },
      ),
      // currentStep is intentionally excluded — the URL is the source of
      // truth for which step the user is on, and the page hook syncs the
      // store from the URL on mount.
      // selectedPlan is also excluded — it's only meaningful while a
      // Stripe Checkout request is in flight (gated by isUpdatingTier in
      // SubscriptionStep). The full-page Stripe redirect doesn't survive
      // in-memory state anyway, so persisting it serves no purpose and
      // would resurface a stale "selected" plan after cancel-and-return.
      partialize: (state) => ({
        role: state.role,
        otherRole: state.otherRole,
        painPoints: state.painPoints,
        otherPainPoint: state.otherPainPoint,
        selectedBilling: state.selectedBilling,
        hasUserSelectedBilling: state.hasUserSelectedBilling,
        selectedCountryCode: state.selectedCountryCode,
        hiredTemplateIds: state.hiredTemplateIds,
      }),
    },
  ),
);
