import { describe, it, expect, beforeEach } from "vitest";
import { useOnboardingWizardStore } from "../store";

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
});

describe("useOnboardingWizardStore", () => {
  describe("initial state", () => {
    it("starts at step 1 with empty fields", () => {
      const state = useOnboardingWizardStore.getState();
      expect(state.currentStep).toBe(1);
      expect(state.name).toBe("");
      expect(state.role).toBe("");
      expect(state.otherRole).toBe("");
      expect(state.painPoints).toEqual([]);
      expect(state.otherPainPoint).toBe("");
    });
  });

  describe("setName", () => {
    it("updates the name", () => {
      useOnboardingWizardStore.getState().setName("Alice");
      expect(useOnboardingWizardStore.getState().name).toBe("Alice");
    });
  });

  describe("setRole", () => {
    it("updates the role", () => {
      useOnboardingWizardStore.getState().setRole("Engineer");
      expect(useOnboardingWizardStore.getState().role).toBe("Engineer");
    });
  });

  describe("setOtherRole", () => {
    it("updates the other role text", () => {
      useOnboardingWizardStore.getState().setOtherRole("Designer");
      expect(useOnboardingWizardStore.getState().otherRole).toBe("Designer");
    });
  });

  describe("togglePainPoint", () => {
    it("adds a pain point", () => {
      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      expect(useOnboardingWizardStore.getState().painPoints).toEqual([
        "slow builds",
      ]);
    });

    it("removes a pain point when toggled again", () => {
      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      expect(useOnboardingWizardStore.getState().painPoints).toEqual([]);
    });

    it("handles multiple pain points", () => {
      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      useOnboardingWizardStore.getState().togglePainPoint("no tests");
      expect(useOnboardingWizardStore.getState().painPoints).toEqual([
        "slow builds",
        "no tests",
      ]);

      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      expect(useOnboardingWizardStore.getState().painPoints).toEqual([
        "no tests",
      ]);
    });

    it("ignores new selections when at the max limit", () => {
      useOnboardingWizardStore.getState().togglePainPoint("a");
      useOnboardingWizardStore.getState().togglePainPoint("b");
      useOnboardingWizardStore.getState().togglePainPoint("c");
      useOnboardingWizardStore.getState().togglePainPoint("d");
      expect(useOnboardingWizardStore.getState().painPoints).toEqual([
        "a",
        "b",
        "c",
      ]);
    });

    it("still allows deselecting when at the max limit", () => {
      useOnboardingWizardStore.getState().togglePainPoint("a");
      useOnboardingWizardStore.getState().togglePainPoint("b");
      useOnboardingWizardStore.getState().togglePainPoint("c");
      useOnboardingWizardStore.getState().togglePainPoint("b");
      expect(useOnboardingWizardStore.getState().painPoints).toEqual([
        "a",
        "c",
      ]);
    });
  });

  describe("setOtherPainPoint", () => {
    it("updates the other pain point text", () => {
      useOnboardingWizardStore.getState().setOtherPainPoint("flaky CI");
      expect(useOnboardingWizardStore.getState().otherPainPoint).toBe(
        "flaky CI",
      );
    });
  });

  describe("nextStep", () => {
    it("increments the step", () => {
      useOnboardingWizardStore.getState().nextStep();
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
    });

    it("clamps at step 4", () => {
      useOnboardingWizardStore.getState().goToStep(4);
      useOnboardingWizardStore.getState().nextStep();
      expect(useOnboardingWizardStore.getState().currentStep).toBe(4);
    });
  });

  describe("prevStep", () => {
    it("decrements the step", () => {
      useOnboardingWizardStore.getState().goToStep(3);
      useOnboardingWizardStore.getState().prevStep();
      expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
    });

    it("clamps at step 1", () => {
      useOnboardingWizardStore.getState().prevStep();
      expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
    });
  });

  describe("goToStep", () => {
    it("jumps to an arbitrary step", () => {
      useOnboardingWizardStore.getState().goToStep(3);
      expect(useOnboardingWizardStore.getState().currentStep).toBe(3);
    });
  });

  describe("reset", () => {
    it("resets all fields to defaults", () => {
      useOnboardingWizardStore.getState().setName("Alice");
      useOnboardingWizardStore.getState().setRole("Engineer");
      useOnboardingWizardStore.getState().setOtherRole("Other");
      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      useOnboardingWizardStore.getState().setOtherPainPoint("flaky CI");
      useOnboardingWizardStore.getState().goToStep(3);

      useOnboardingWizardStore.getState().reset();

      const state = useOnboardingWizardStore.getState();
      expect(state.currentStep).toBe(1);
      expect(state.name).toBe("");
      expect(state.role).toBe("");
      expect(state.otherRole).toBe("");
      expect(state.painPoints).toEqual([]);
      expect(state.otherPainPoint).toBe("");
    });
  });
});
