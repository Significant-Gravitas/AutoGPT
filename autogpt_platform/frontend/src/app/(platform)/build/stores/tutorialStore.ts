import { create } from "zustand";

type TutorialStore = {
  isTutorialRunning: boolean;
  setIsTutorialRunning: (isTutorialRunning: boolean) => void;

  currentStep: number;
  setCurrentStep: (currentStep: number) => void;
};

export const useTutorialStore = create<TutorialStore>((set) => ({
  isTutorialRunning: false,
  setIsTutorialRunning: (isTutorialRunning) => set({ isTutorialRunning }),

  currentStep: 0,
  setCurrentStep: (currentStep) => set({ currentStep }),
}));
