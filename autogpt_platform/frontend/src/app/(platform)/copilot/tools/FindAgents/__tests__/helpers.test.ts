import { describe, expect, it } from "vitest";
import { getAnimationText, getSourceLabelFromToolType } from "../helpers";

describe("FindAgents getSourceLabelFromToolType", () => {
  it("labels an unknown tool as workflows", () => {
    expect(getSourceLabelFromToolType(undefined)).toEqual({
      source: "unknown",
      label: "Workflows",
    });
  });
});

describe("FindAgents getAnimationText", () => {
  it("says it is searching the library", () => {
    expect(
      getAnimationText({
        type: "tool-find_library_agent",
        state: "input-available",
        input: { query: "digest" },
      }),
    ).toBe('Looking for library workflows matching "digest"');
  });

  it("says it is searching the marketplace", () => {
    expect(
      getAnimationText({ type: "tool-find_agent", state: "input-streaming" }),
    ).toBe("Looking for workflows in the marketplace");
  });

  it("reports no results", () => {
    expect(
      getAnimationText({
        type: "tool-find_agent",
        state: "output-available",
        input: { query: "digest" },
        output: { type: "no_results", message: "none", suggestions: [] },
      }),
    ).toBe('No workflows found matching "digest"');
  });

  it("counts the workflows found", () => {
    expect(
      getAnimationText({
        type: "tool-find_library_agent",
        state: "output-available",
        output: { type: "agents_found", agents: [], count: 1 },
      }),
    ).toBe("Found 1 workflow");
    expect(
      getAnimationText({
        type: "tool-find_library_agent",
        state: "output-available",
        output: { type: "agents_found", agents: [], count: 3 },
      }),
    ).toBe("Found 3 workflows");
  });

  it("reports an error response", () => {
    expect(
      getAnimationText({
        type: "tool-find_agent",
        state: "output-available",
        input: { query: "digest" },
        output: { type: "error", error: "boom", message: "boom" },
      }),
    ).toBe('Error finding workflows matching "digest"');
  });

  it("reports a tool error", () => {
    expect(
      getAnimationText({ type: "tool-find_agent", state: "output-error" }),
    ).toBe("Error finding workflows");
  });
});
