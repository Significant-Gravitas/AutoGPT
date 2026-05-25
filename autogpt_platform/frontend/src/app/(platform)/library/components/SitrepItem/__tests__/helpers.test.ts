import type { SitrepItemData, SitrepPriority } from "../../../types";
import { describe, expect, it } from "vitest";
import { buildAutoPilotPrompt } from "../helpers";

function makeItem(
  priority: SitrepPriority,
  overrides: Partial<SitrepItemData> = {},
): SitrepItemData {
  return {
    id: "1",
    agentID: "a-1",
    agentName: "Weather Bot",
    priority,
    message: "Something happened",
    status: "idle",
    ...overrides,
  };
}

describe("buildAutoPilotPrompt", () => {
  it("asks to investigate logs and quotes the error message", () => {
    const prompt = buildAutoPilotPrompt(
      makeItem("error", { message: "Boom at step 3" }),
    );
    expect(prompt).toContain("Weather Bot");
    expect(prompt).toContain("Boom at step 3");
    expect(prompt.toLowerCase()).toContain("logs");
  });

  it("asks for a status update on running agents", () => {
    expect(buildAutoPilotPrompt(makeItem("running"))).toContain(
      "status update",
    );
  });

  it("offers to keep, update, or re-run stale agents", () => {
    expect(buildAutoPilotPrompt(makeItem("stale"))).toContain(
      "hasn't run recently",
    );
  });

  it("summarizes results for successful runs", () => {
    expect(buildAutoPilotPrompt(makeItem("success")).toLowerCase()).toContain(
      "summarize",
    );
  });

  it("summarizes the trigger configuration for listening agents", () => {
    expect(buildAutoPilotPrompt(makeItem("listening")).toLowerCase()).toContain(
      "listening for",
    );
  });

  it("asks when a scheduled agent runs next", () => {
    expect(buildAutoPilotPrompt(makeItem("scheduled")).toLowerCase()).toContain(
      "scheduled to run next",
    );
  });

  it("offers to keep, update, or re-run idle agents", () => {
    expect(buildAutoPilotPrompt(makeItem("idle"))).toContain("has been idle");
  });
});
