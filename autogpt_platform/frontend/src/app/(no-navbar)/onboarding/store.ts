import { create } from "zustand";

export const MAX_PAIN_POINT_SELECTIONS = 3;
export type Step = 1 | 2 | 3 | 4 | 5;

interface OnboardingWizardState {
  currentStep: Step;
  name: string;
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
  selectedPlan: string | null;
  selectedBilling: "monthly" | "yearly";
  selectedCountryCode: string;
  setName(name: string): void;
  setRole(role: string): void;
  setOtherRole(otherRole: string): void;
  togglePainPoint(painPoint: string): void;
  setOtherPainPoint(otherPainPoint: string): void;
  setSelectedPlan(plan: string): void;
  setSelectedBilling(billing: "monthly" | "yearly"): void;
  setSelectedCountryCode(code: string): void;
  nextStep(): void;
  prevStep(): void;
  goToStep(step: Step): void;
  reset(): void;
}

export const useOnboardingWizardStore = create<OnboardingWizardState>(
  (set) => ({
    currentStep: 1,
    name: "",
    role: "",
    otherRole: "",
    painPoints: [],
    otherPainPoint: "",
    selectedPlan: null,
    selectedBilling: "monthly",
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
      set({ selectedBilling: billing });
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
        selectedCountryCode: "US",
      });
    },
  }),
);
