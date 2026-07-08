import { describe, expect, it } from "vitest";
import {
  getPhaseLabel,
  isPlanningInFlight,
  parsePlanPartData,
  shortModelName,
} from "../helpers";

describe("parsePlanPartData", () => {
  it("returns null for non-object input", () => {
    expect(parsePlanPartData(null)).toBeNull();
    expect(parsePlanPartData("nope")).toBeNull();
    expect(parsePlanPartData(42)).toBeNull();
  });

  it("returns null when phase is missing", () => {
    expect(parsePlanPartData({ steps: [] })).toBeNull();
  });

  it("parses a planned payload with steps", () => {
    const data = parsePlanPartData({
      phase: "planned",
      steps: [
        {
          id: "step-1",
          description: "Fetch issues",
          expectedTools: ["github_search_issues", 3],
          successCriteria: "Issues retrieved",
        },
      ],
      plannerModel: "anthropic/claude-opus-4.7",
      executorModel: "anthropic/claude-sonnet-4-6",
      revision: 0,
      reason: null,
      executorPrompt: "<execution_plan>…</execution_plan>",
    });

    expect(data).not.toBeNull();
    expect(data!.phase).toBe("planned");
    expect(data!.steps).toHaveLength(1);
    // Non-string tool entries are dropped, not coerced.
    expect(data!.steps[0].expectedTools).toEqual(["github_search_issues"]);
    expect(data!.executorPrompt).toContain("execution_plan");
  });

  it("defaults missing optional fields", () => {
    const data = parsePlanPartData({ phase: "planning" });
    expect(data).not.toBeNull();
    expect(data!.steps).toEqual([]);
    expect(data!.plannerModel).toBeNull();
    expect(data!.revision).toBe(0);
  });
});

describe("getPhaseLabel", () => {
  const base = {
    steps: [],
    plannerModel: null,
    executorModel: null,
    revision: 0,
    reason: null,
    executorPrompt: null,
  };

  it("shows the step count for a planned card", () => {
    expect(
      getPhaseLabel({
        ...base,
        phase: "planned",
        steps: [
          { id: "a", description: "x", expectedTools: [], successCriteria: "" },
          { id: "b", description: "y", expectedTools: [], successCriteria: "" },
        ],
      }),
    ).toBe("Task plan · 2 steps");
  });

  it("singularises a one-step plan", () => {
    expect(
      getPhaseLabel({
        ...base,
        phase: "planned",
        steps: [
          { id: "a", description: "x", expectedTools: [], successCriteria: "" },
        ],
      }),
    ).toBe("Task plan · 1 step");
  });

  it("labels a revision with its version number", () => {
    expect(
      getPhaseLabel({ ...base, phase: "replanned", revision: 1 }),
    ).toBe("Plan revised (v2)");
  });

  it("uses the static label for the planning phase", () => {
    expect(getPhaseLabel({ ...base, phase: "planning" })).toBe(
      "Planning your task…",
    );
  });
});

describe("helpers misc", () => {
  it("isPlanningInFlight is true only for the planning phase", () => {
    const base = {
      steps: [],
      plannerModel: null,
      executorModel: null,
      revision: 0,
      reason: null,
      executorPrompt: null,
    };
    expect(isPlanningInFlight({ ...base, phase: "planning" })).toBe(true);
    expect(isPlanningInFlight({ ...base, phase: "planned" })).toBe(false);
  });

  it("shortModelName strips the provider prefix", () => {
    expect(shortModelName("anthropic/claude-opus-4.7")).toBe(
      "claude-opus-4.7",
    );
    expect(shortModelName("plainmodel")).toBe("plainmodel");
    expect(shortModelName(null)).toBeNull();
  });
});
