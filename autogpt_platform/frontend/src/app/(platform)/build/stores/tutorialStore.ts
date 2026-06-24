import { create } from "zustand";

type TutorialStore = {
  isTutorialRunning: boolean;
  setIsTutorialRunning: (isTutorialRunning: boolean) => void;

  currentStep: number;
  setCurrentStep: (currentStep: number) => void;

  // Force open the run input dialog from the tutorial
  forceOpenRunInputDialog: boolean;
  setForceOpenRunInputDialog: (forceOpen: boolean) => void;

  // Track input values filled in the dialog
  tutorialInputValues: Record<string, any>;
  setTutorialInputValues: (values: Record<string, any>) => void;
};

export const useTutorialStore = create<TutorialStore>((set) => ({
  isTutorialRunning: false,
  setIsTutorialRunning: (isTutorialRunning) => set({ isTutorialRunning }),

  currentStep: 0,
  setCurrentStep: (currentStep) => set({ currentStep }),

  forceOpenRunInputDialog: false,
  setForceOpenRunInputDialog: (forceOpen) =>
    set({ forceOpenRunInputDialog: forceOpen }),

  tutorialInputValues: {},
  setTutorialInputValues: (values) => set({ tutorialInputValues: values }),
}));
