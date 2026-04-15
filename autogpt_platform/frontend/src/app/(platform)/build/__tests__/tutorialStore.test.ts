import { describe, it, expect, beforeEach } from "vitest";
import { useTutorialStore } from "../stores/tutorialStore";

beforeEach(() => {
  useTutorialStore.setState({
    isTutorialRunning: false,
    currentStep: 0,
    forceOpenRunInputDialog: false,
    tutorialInputValues: {},
  });
});

describe("tutorialStore", () => {
  describe("initial state", () => {
    it("starts with tutorial not running at step 0", () => {
      const state = useTutorialStore.getState();
      expect(state.isTutorialRunning).toBe(false);
      expect(state.currentStep).toBe(0);
      expect(state.forceOpenRunInputDialog).toBe(false);
      expect(state.tutorialInputValues).toEqual({});
    });
  });

  describe("setIsTutorialRunning", () => {
    it("starts the tutorial", () => {
      useTutorialStore.getState().setIsTutorialRunning(true);
      expect(useTutorialStore.getState().isTutorialRunning).toBe(true);
    });

    it("stops the tutorial", () => {
      useTutorialStore.getState().setIsTutorialRunning(true);
      useTutorialStore.getState().setIsTutorialRunning(false);
      expect(useTutorialStore.getState().isTutorialRunning).toBe(false);
    });
  });

  describe("setCurrentStep", () => {
    it("advances to a step", () => {
      useTutorialStore.getState().setCurrentStep(3);
      expect(useTutorialStore.getState().currentStep).toBe(3);
    });

    it("can go back to a previous step", () => {
      useTutorialStore.getState().setCurrentStep(5);
      useTutorialStore.getState().setCurrentStep(2);
      expect(useTutorialStore.getState().currentStep).toBe(2);
    });

    it("can reset to step 0", () => {
      useTutorialStore.getState().setCurrentStep(4);
      useTutorialStore.getState().setCurrentStep(0);
      expect(useTutorialStore.getState().currentStep).toBe(0);
    });
  });

  describe("setForceOpenRunInputDialog", () => {
    it("forces the dialog open", () => {
      useTutorialStore.getState().setForceOpenRunInputDialog(true);
      expect(useTutorialStore.getState().forceOpenRunInputDialog).toBe(true);
    });

    it("closes the forced dialog", () => {
      useTutorialStore.getState().setForceOpenRunInputDialog(true);
      useTutorialStore.getState().setForceOpenRunInputDialog(false);
      expect(useTutorialStore.getState().forceOpenRunInputDialog).toBe(false);
    });
  });

  describe("setTutorialInputValues", () => {
    it("sets input values", () => {
      useTutorialStore
        .getState()
        .setTutorialInputValues({ topic: "AI agents" });
      expect(useTutorialStore.getState().tutorialInputValues).toEqual({
        topic: "AI agents",
      });
    });

    it("replaces previous values entirely", () => {
      useTutorialStore.getState().setTutorialInputValues({ a: "1" });
      useTutorialStore.getState().setTutorialInputValues({ b: "2" });
      expect(useTutorialStore.getState().tutorialInputValues).toEqual({
        b: "2",
      });
    });

    it("clears values with empty object", () => {
      useTutorialStore.getState().setTutorialInputValues({ x: "y" });
      useTutorialStore.getState().setTutorialInputValues({});
      expect(useTutorialStore.getState().tutorialInputValues).toEqual({});
    });
  });

  describe("step progression lifecycle", () => {
    it("simulates a full tutorial run", () => {
      useTutorialStore.getState().setIsTutorialRunning(true);
      expect(useTutorialStore.getState().isTutorialRunning).toBe(true);

      useTutorialStore.getState().setCurrentStep(1);
      useTutorialStore.getState().setCurrentStep(2);
      useTutorialStore.getState().setCurrentStep(3);

      useTutorialStore.getState().setForceOpenRunInputDialog(true);
      useTutorialStore
        .getState()
        .setTutorialInputValues({ prompt: "test prompt" });
      useTutorialStore.getState().setForceOpenRunInputDialog(false);

      useTutorialStore.getState().setCurrentStep(4);
      useTutorialStore.getState().setIsTutorialRunning(false);
      useTutorialStore.getState().setCurrentStep(0);

      const state = useTutorialStore.getState();
      expect(state.isTutorialRunning).toBe(false);
      expect(state.currentStep).toBe(0);
    });
  });
});
