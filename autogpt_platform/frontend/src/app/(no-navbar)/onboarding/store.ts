import { create } from "zustand";

export type Step = 1 | 2 | 3 | 4;

interface OnboardingWizardState {
  currentStep: Step;
  name: string;
  role: string;
  otherRole: string;
  painPoints: string[];
  otherPainPoint: string;
  setName(name: string): void;
  setRole(role: string): void;
  setOtherRole(otherRole: string): void;
  togglePainPoint(painPoint: string): void;
  setOtherPainPoint(otherPainPoint: string): void;
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
    nextStep() {
      set((state) => ({
        currentStep: Math.min(4, state.currentStep + 1) as Step,
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
      });
    },
  }),
);
