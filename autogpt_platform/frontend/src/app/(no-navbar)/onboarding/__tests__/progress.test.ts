import { describe, expect, it } from "vitest";
import { readProgress, resumeStep } from "../progress";
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
