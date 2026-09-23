import { describe, it, expect, beforeEach } from "vitest";
import { buildStepLayout, useOnboardingWizardStore } from "../store";

beforeEach(() => {
  useOnboardingWizardStore.getState().reset();
});

describe("useOnboardingWizardStore", () => {
  describe("initial state", () => {
    it("starts at step 1 with empty fields", () => {
      const state = useOnboardingWizardStore.getState();
      expect(state.currentStep).toBe(1);
      expect(state.role).toBe("");
      expect(state.otherRole).toBe("");
      expect(state.painPoints).toEqual([]);
      expect(state.otherPainPoint).toBe("");
    });

    it("defaults to monthly billing", () => {
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "monthly",
      );
      expect(useOnboardingWizardStore.getState().hasUserSelectedBilling).toBe(
        false,
      );
    });
  });

  describe("billing selection", () => {
    it("tracks when the billing cycle was selected by the user", () => {
      useOnboardingWizardStore.getState().setSelectedBilling("monthly");

      const state = useOnboardingWizardStore.getState();
      expect(state.selectedBilling).toBe("monthly");
      expect(state.hasUserSelectedBilling).toBe(true);
    });

    it("applies experiment billing until the user has selected a cycle", () => {
      useOnboardingWizardStore
        .getState()
        .applyPricingExperimentBilling("monthly");
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "monthly",
      );

      useOnboardingWizardStore.getState().setSelectedBilling("yearly");
      useOnboardingWizardStore
        .getState()
        .applyPricingExperimentBilling("monthly");

      const state = useOnboardingWizardStore.getState();
      expect(state.selectedBilling).toBe("yearly");
      expect(state.hasUserSelectedBilling).toBe(true);
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

    it("clamps at the last step", () => {
      useOnboardingWizardStore.getState().goToStep(7);
      useOnboardingWizardStore.getState().nextStep();
      expect(useOnboardingWizardStore.getState().currentStep).toBe(7);
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

  describe("markHired", () => {
    it("records each hired template once", () => {
      useOnboardingWizardStore.getState().markHired("tpl-maria");
      useOnboardingWizardStore.getState().markHired("tpl-maria");
      useOnboardingWizardStore.getState().markHired("tpl-max");
      expect(useOnboardingWizardStore.getState().hiredTemplateIds).toEqual([
        "tpl-maria",
        "tpl-max",
      ]);
    });
  });

  describe("reset", () => {
    it("resets all fields to defaults", () => {
      useOnboardingWizardStore.getState().setRole("Engineer");
      useOnboardingWizardStore.getState().setOtherRole("Other");
      useOnboardingWizardStore.getState().togglePainPoint("slow builds");
      useOnboardingWizardStore.getState().setOtherPainPoint("flaky CI");
      useOnboardingWizardStore.getState().markHired("tpl-maria");
      useOnboardingWizardStore.getState().goToStep(3);

      useOnboardingWizardStore.getState().reset();

      const state = useOnboardingWizardStore.getState();
      expect(state.currentStep).toBe(1);
      expect(state.role).toBe("");
      expect(state.otherRole).toBe("");
      expect(state.painPoints).toEqual([]);
      expect(state.otherPainPoint).toBe("");
      expect(state.hiredTemplateIds).toEqual([]);
    });
  });
});

describe("buildStepLayout", () => {
  it("leads with the paywall and slots the hire step after the brain dump", () => {
    expect(
      buildStepLayout({ hasIntro: true, hasHire: true, hasPaywall: true }),
    ).toEqual({
      subscription: 1,
      team: 2,
      autopilot: 3,
      role: 4,
      painPoints: 5,
      hire: 6,
      preparing: 7,
    });
  });

  it("leaves the hire step out when it is off", () => {
    expect(buildStepLayout({ hasIntro: true, hasPaywall: true })).toEqual({
      subscription: 1,
      team: 2,
      autopilot: 3,
      role: 4,
      painPoints: 5,
      preparing: 6,
    });
  });

  it("closes with the connection on self-host, never beside a paywall", () => {
    expect(buildStepLayout({ hasConnect: true })).toEqual({
      role: 1,
      painPoints: 2,
      connect: 3,
      preparing: 4,
    });
    expect(
      buildStepLayout({ hasPaywall: true, hasConnect: true }).connect,
    ).toBeUndefined();
  });
});
