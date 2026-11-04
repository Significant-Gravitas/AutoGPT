import { describe, expect, it } from "vitest";
import { getAnimationText } from "../helpers";

describe("ViewAgentOutput getAnimationText", () => {
  it("names a library workflow while its output loads", () => {
    expect(
      getAnimationText({
        state: "input-available",
        input: { library_agent_id: "lib-1" },
      }),
    ).toBe('Retrieving workflow output "Library workflow lib-1"');
  });

  it("keeps the loading text until the output can be read", () => {
    expect(
      getAnimationText({
        state: "output-available",
        input: { agent_name: "Digest" },
        output: "",
      }),
    ).toBe('Retrieving workflow output "Digest"');
  });

  it("reports a loaded output with no execution", () => {
    expect(
      getAnimationText({
        state: "output-available",
        output: { type: "agent_output", agent_id: "a1", agent_name: "Digest" },
      }),
    ).toBe("Retrieved workflow output");
  });

  it("reports an error response as a failed load", () => {
    expect(
      getAnimationText({
        state: "output-available",
        output: { type: "error", error: "boom", message: "boom" },
      }),
    ).toBe("Error loading workflow output");
  });

  it("reports a tool error as a failed load", () => {
    expect(getAnimationText({ state: "output-error" })).toBe(
      "Error loading workflow output",
    );
  });

  it("falls back to the plain loading text", () => {
    expect(getAnimationText({ state: "approval-requested" })).toBe(
      "Retrieving workflow output",
    );
  });
});
