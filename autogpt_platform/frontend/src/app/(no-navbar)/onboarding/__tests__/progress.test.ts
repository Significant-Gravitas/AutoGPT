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
  it("resumes after an optional hire step disappears", () => {
    expect(
      resumeStep({
        progress: makeProgress({
          currentStep: "hire",
          completedSteps: ["role", "painPoints"],
        }),
        steps: buildStepLayout({ hasPaywall: true }),
      }),
    ).toBe(3);
  });
  it("honors a previous-step URL without allowing an incomplete step to be skipped", () => {
    const progress = makeProgress({
      currentStep: "subscription",
      completedSteps: ["role", "painPoints"],
    });
    const steps = buildStepLayout({ hasPaywall: true });
    expect(resumeStep({ progress, steps, requestedStep: "role" })).toBe(1);
    expect(resumeStep({ progress, steps, requestedStep: "preparing" })).toBe(3);
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
