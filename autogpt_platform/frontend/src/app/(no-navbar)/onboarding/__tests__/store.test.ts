import { describe, expect, it, beforeEach } from "vitest";
import { useOnboardingWizardStore } from "../store";

describe("useOnboardingWizardStore", () => {
  beforeEach(() => {
    useOnboardingWizardStore.getState().reset();
  });

  it("starts at step 1 with empty state", () => {
    const state = useOnboardingWizardStore.getState();
    expect(state.currentStep).toBe(1);
    expect(state.name).toBe("");
    expect(state.role).toBe("");
    expect(state.otherRole).toBe("");
    expect(state.painPoints).toEqual([]);
    expect(state.otherPainPoint).toBe("");
  });

  it("setName updates name", () => {
    useOnboardingWizardStore.getState().setName("Alice");
    expect(useOnboardingWizardStore.getState().name).toBe("Alice");
  });

  it("setRole updates role", () => {
    useOnboardingWizardStore.getState().setRole("developer");
    expect(useOnboardingWizardStore.getState().role).toBe("developer");
  });

  it("setOtherRole updates otherRole", () => {
    useOnboardingWizardStore.getState().setOtherRole("consultant");
    expect(useOnboardingWizardStore.getState().otherRole).toBe("consultant");
  });

  it("togglePainPoint adds a pain point when not present", () => {
    useOnboardingWizardStore.getState().togglePainPoint("slow builds");
    expect(useOnboardingWizardStore.getState().painPoints).toEqual([
      "slow builds",
    ]);
  });

  it("togglePainPoint removes a pain point when already present", () => {
    const store = useOnboardingWizardStore.getState();
    store.togglePainPoint("slow builds");
    store.togglePainPoint("slow builds");
    expect(useOnboardingWizardStore.getState().painPoints).toEqual([]);
  });

  it("togglePainPoint handles multiple pain points", () => {
    const store = useOnboardingWizardStore.getState();
    store.togglePainPoint("slow builds");
    store.togglePainPoint("deployment");
    expect(useOnboardingWizardStore.getState().painPoints).toEqual([
      "slow builds",
      "deployment",
    ]);
    store.togglePainPoint("slow builds");
    expect(useOnboardingWizardStore.getState().painPoints).toEqual([
      "deployment",
    ]);
  });

  it("setOtherPainPoint updates otherPainPoint", () => {
    useOnboardingWizardStore.getState().setOtherPainPoint("custom issue");
    expect(useOnboardingWizardStore.getState().otherPainPoint).toBe(
      "custom issue",
    );
  });

  it("nextStep increments currentStep", () => {
    useOnboardingWizardStore.getState().nextStep();
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  it("nextStep does not exceed 4", () => {
    const store = useOnboardingWizardStore.getState();
    store.goToStep(4);
    store.nextStep();
    expect(useOnboardingWizardStore.getState().currentStep).toBe(4);
  });

  it("prevStep decrements currentStep", () => {
    const store = useOnboardingWizardStore.getState();
    store.nextStep();
    store.prevStep();
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });

  it("prevStep does not go below 1", () => {
    useOnboardingWizardStore.getState().prevStep();
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });

  it("goToStep sets currentStep directly", () => {
    useOnboardingWizardStore.getState().goToStep(3);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(3);
  });

  it("reset restores all values to defaults", () => {
    const store = useOnboardingWizardStore.getState();
    store.setName("Alice");
    store.setRole("developer");
    store.nextStep();
    store.togglePainPoint("slow builds");
    store.setOtherPainPoint("other");
    store.reset();

    const state = useOnboardingWizardStore.getState();
    expect(state.currentStep).toBe(1);
    expect(state.name).toBe("");
    expect(state.role).toBe("");
    expect(state.otherRole).toBe("");
    expect(state.painPoints).toEqual([]);
    expect(state.otherPainPoint).toBe("");
  });
});
