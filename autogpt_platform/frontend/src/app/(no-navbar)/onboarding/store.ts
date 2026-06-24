import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

export const MAX_PAIN_POINT_SELECTIONS = 3;
export type Step = 1 | 2 | 3 | 4 | 5;

// Wizard step layouts. With payments enabled the paywall is the FIRST step so
// the user pays before personalising; the profile-collection steps shift down
// by one. Centralised here so page rendering, the page hook's URL clamping and
// the SubscriptionStep's Stripe success/cancel return URLs can't drift apart.
export const PAYWALL_FIRST_STEPS = {
  subscription: 1,
  welcome: 2,
  role: 3,
  painPoints: 4,
  preparing: 5,
} as const;

export const NO_PAYWALL_STEPS = {
  welcome: 1,
  role: 2,
  painPoints: 3,
  preparing: 4,
} as const;

interface OnboardingWizardState {
  currentStep: Step;
  name: string;
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
  selectedPlan: string | null;
  selectedBilling: "monthly" | "yearly";
  hasUserSelectedBilling: boolean;
  selectedCountryCode: string;
  setName(name: string): void;
  setRole(role: string): void;
  setOtherRole(otherRole: string): void;
  togglePainPoint(painPoint: string): void;
  setOtherPainPoint(otherPainPoint: string): void;
  setSelectedPlan(plan: string): void;
  setSelectedBilling(billing: "monthly" | "yearly"): void;
  applyPricingExperimentBilling(billing: "monthly" | "yearly"): void;
  setSelectedCountryCode(code: string): void;
  nextStep(): void;
  prevStep(): void;
  goToStep(step: Step): void;
  reset(): void;
}

export const useOnboardingWizardStore = create<OnboardingWizardState>()(
  persist(
    (set) => ({
      currentStep: 1,
      name: "",
      role: "",
      otherRole: "",
      painPoints: [],
      otherPainPoint: "",
      selectedPlan: null,
      selectedBilling: "monthly",
      hasUserSelectedBilling: false,
      selectedCountryCode: "US",
      setName(name) {
        set({ name });
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
      nextStep() {
        set((state) => ({
          currentStep: Math.min(5, state.currentStep + 1) as Step,
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
          name: "",
          role: "",
          otherRole: "",
          painPoints: [],
          otherPainPoint: "",
          selectedPlan: null,
          selectedBilling: "monthly",
          hasUserSelectedBilling: false,
          selectedCountryCode: "US",
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
        name: state.name,
        role: state.role,
        otherRole: state.otherRole,
        painPoints: state.painPoints,
        otherPainPoint: state.otherPainPoint,
        selectedBilling: state.selectedBilling,
        hasUserSelectedBilling: state.hasUserSelectedBilling,
        selectedCountryCode: state.selectedCountryCode,
      }),
    },
  ),
);
