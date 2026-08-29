import { describe, expect, it } from "vitest";
import type { ToolUIPart } from "ai";
import { getAnimationText } from "../helpers";

type Part = {
  state: ToolUIPart["state"];
  input?: unknown;
  output?: unknown;
};

const LOADING = "Creating agent, this might take a minute";

describe("CreateAgent getAnimationText", () => {
  it("returns the loading copy while streaming input", () => {
    expect(getAnimationText({ state: "input-streaming" })).toBe(LOADING);
    expect(getAnimationText({ state: "input-available" })).toBe(LOADING);
  });

  it("returns the loading copy when output is unparseable", () => {
    expect(
      getAnimationText({ state: "output-available", output: "not json" }),
    ).toBe(LOADING);
  });

  it("falls back to the loading copy for unknown states", () => {
    expect(
      getAnimationText({ state: "partial-output" } as unknown as Part),
    ).toBe(LOADING);
  });

  it("describes recognized outputs", () => {
    expect(
      getAnimationText({
        state: "output-available",
        output: { type: "agent_builder_saved", agent_name: "Bot" },
      }),
    ).toBe("Saved Bot");
    expect(
      getAnimationText({
        state: "output-available",
        output: { type: "agent_builder_preview", agent_name: "Bot" },
      }),
    ).toBe('Preview "Bot"');
    expect(
      getAnimationText({
        state: "output-available",
        output: { type: "suggested_goal", suggested_goal: "x" },
      }),
    ).toBe("Goal needs refinement");
  });

  it("returns the error copy on error states", () => {
    expect(getAnimationText({ state: "output-error" })).toBe(
      "Error creating agent",
    );
  });
});
