import { describe, expect, it } from "vitest";
import { placeToSave, readProgress, resumeStep } from "../progress";
import { buildStepLayout } from "../store";
import { makeProgress } from "./progress-fixture";

it.each([
  null,
  {},
  { version: 0 },
  { currentStep: 7 },
  { ...makeProgress(), completedSteps: [7] },
])("rejects malformed or old numeric drafts", (value) => {
  expect(readProgress(value)).toBeNull();
});

describe("semantic progress", () => {
  it("resumes a draft saved on the retired hire step at the plan step", () => {
    const steps = buildStepLayout({ hasPaywall: true });
    for (const completedSteps of [
      ["team", "autopilot", "role", "painPoints"],
      ["team", "autopilot", "role", "painPoints", "hire"],
    ] as const) {
      const progress = readProgress(
        makeProgress({
          currentStep: "hire",
          completedSteps: [...completedSteps],
        }),
      );
      expect(progress).not.toBeNull();
      expect(resumeStep({ progress, steps })).toBe(steps.subscription);
    }
  });

  it("keeps a plan-only step done once it was left, but never a paywall", () => {
    const progress = makeProgress({
      currentStep: "connect",
      completedSteps: [
        "team",
        "autopilot",
        "role",
        "painPoints",
        "subscription",
      ],
    });
    const selfHosted = buildStepLayout({ hasPlan: true, hasConnect: true });
    expect(resumeStep({ progress, steps: selfHosted, hasPaywall: false })).toBe(
      selfHosted.connect,
    );
    const paywall = buildStepLayout({ hasPaywall: true });
    expect(resumeStep({ progress, steps: paywall })).toBe(paywall.subscription);
  });
  it("resumes a draft saved on the interview where the layout has it, and at the brain dump where it does not", () => {
    const progress = readProgress(
      makeProgress({
        currentStep: "interview",
        completedSteps: ["team", "autopilot", "role"],
      }),
    );
    expect(progress).not.toBeNull();

    const withInterview = buildStepLayout({
      hasPaywall: true,
      hasInterview: true,
    });
    expect(resumeStep({ progress, steps: withInterview })).toBe(
      withInterview.interview,
    );
    const withBrainDump = buildStepLayout({ hasPaywall: true });
    expect(resumeStep({ progress, steps: withBrainDump })).toBe(
      withBrainDump.painPoints,
    );
  });
  it("resumes past the brain dump once the interview was done", () => {
    const progress = readProgress(
      makeProgress({
        currentStep: "subscription",
        completedSteps: ["team", "autopilot", "role", "interview"],
      }),
    );
    const steps = buildStepLayout({ hasPaywall: true });

    expect(resumeStep({ progress, steps })).toBe(steps.subscription);
  });

  it("resumes past the interview once the brain dump was done", () => {
    const progress = readProgress(
      makeProgress({
        currentStep: "subscription",
        completedSteps: ["team", "autopilot", "role", "painPoints"],
      }),
    );
    const steps = buildStepLayout({ hasPaywall: true, hasInterview: true });

    expect(resumeStep({ progress, steps })).toBe(steps.subscription);
  });

  it("resumes a draft saved on the interview on the brain dump when the layout has only that", () => {
    const progress = readProgress(
      makeProgress({
        currentStep: "interview",
        completedSteps: ["team", "autopilot", "role", "interview"],
      }),
    );
    const steps = buildStepLayout({ hasPaywall: true });

    expect(resumeStep({ progress, steps })).toBe(steps.painPoints);
  });

  it("resumes a draft saved on the brain dump on the interview when the layout has only that", () => {
    const progress = readProgress(
      makeProgress({
        currentStep: "painPoints",
        completedSteps: ["team", "autopilot", "role"],
      }),
    );
    const steps = buildStepLayout({ hasPaywall: true, hasInterview: true });

    expect(resumeStep({ progress, steps })).toBe(steps.interview);
  });

  it("saves a place on the brain dump as the interview's on a layout that left the interview out", () => {
    const withoutInterview = buildStepLayout({ hasPaywall: true });
    const onBrainDump = makeProgress({
      currentStep: "painPoints",
      completedSteps: ["team", "autopilot", "role"],
    });

    expect(placeToSave(onBrainDump, withoutInterview)).toEqual({
      ...onBrainDump,
      currentStep: "interview",
    });
    // Saved so, the next load lays out either step on that place.
    const saved = placeToSave(onBrainDump, withoutInterview);
    const withInterview = buildStepLayout({
      hasPaywall: true,
      hasInterview: true,
    });
    expect(resumeStep({ progress: saved, steps: withInterview })).toBe(
      withInterview.interview,
    );
    expect(resumeStep({ progress: saved, steps: withoutInterview })).toBe(
      withoutInterview.painPoints,
    );
  });

  it("saves every other place as it is", () => {
    const steps = buildStepLayout({ hasPaywall: true });
    for (const currentStep of ["role", "subscription"] as const) {
      const progress = makeProgress({ currentStep });
      expect(placeToSave(progress, steps)).toBe(progress);
    }
  });

  it("honors a previous-step URL without allowing an incomplete step to be skipped", () => {
    const progress = makeProgress({
      currentStep: "subscription",
      completedSteps: ["team", "autopilot", "role", "painPoints"],
    });
    const steps = buildStepLayout({ hasPaywall: true });
    expect(resumeStep({ progress, steps, requestedStep: "role" })).toBe(3);
    expect(resumeStep({ progress, steps, requestedStep: "preparing" })).toBe(5);
  });
});

it.each([
  { painPoints: [""] },
  { hiredTemplateIds: [""] },
  { role: "bad\u0000input" },
  { otherRole: "x".repeat(101) },
])("rejects drafts the API cannot save", (fields) => {
  expect(readProgress(makeProgress(fields))).toBeNull();
});
