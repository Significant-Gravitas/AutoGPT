/**
 * Unit tests for DecomposeGoal/helpers.tsx
 *
 * Covers: parseOutput / getDecomposeGoalOutput, type guards, getAnimationText
 */

import { describe, expect, it } from "vitest";
import {
  getAnimationText,
  getDecomposeGoalOutput,
  isDecompositionOutput,
  isErrorOutput,
  type DecomposeErrorOutput,
  type DecomposeGoalOutput,
  type TaskDecompositionOutput,
} from "../helpers";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

const DECOMPOSITION: TaskDecompositionOutput = {
  type: "task_decomposition",
  message: "Here's the plan (3 steps):",
  goal: "Build a news summarizer",
  steps: [
    {
      step_id: "step_1",
      description: "Accept a topic from the user",
      action: "add_input",
      block_name: null,
      status: "pending",
    },
    {
      step_id: "step_2",
      description: "Summarize the topic with AI",
      action: "add_block",
      block_name: "AI Text Generator",
      status: "pending",
    },
    {
      step_id: "step_3",
      description: "Hand the result back to the user",
      action: "connect_blocks",
      block_name: null,
      status: "pending",
    },
  ],
  step_count: 3,
};

const ERROR_OUTPUT: DecomposeErrorOutput = {
  type: "error",
  error: "missing_steps",
  message: "Please provide at least one step.",
};

// ---------------------------------------------------------------------------
// isDecompositionOutput
// ---------------------------------------------------------------------------

describe("isDecompositionOutput", () => {
  it("returns true for a full decomposition output", () => {
    expect(isDecompositionOutput(DECOMPOSITION)).toBe(true);
  });

  it("returns false for an error output", () => {
    expect(
      isDecompositionOutput(ERROR_OUTPUT as unknown as DecomposeGoalOutput),
    ).toBe(false);
  });

  it("returns false when steps is not an array (type guard tightness)", () => {
    const malformed = {
      steps: "not-an-array",
      goal: "test",
    } as unknown as DecomposeGoalOutput;
    expect(isDecompositionOutput(malformed)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// isErrorOutput
// ---------------------------------------------------------------------------

describe("isErrorOutput", () => {
  it("returns true for error output", () => {
    expect(isErrorOutput(ERROR_OUTPUT as unknown as DecomposeGoalOutput)).toBe(
      true,
    );
  });

  it("returns false for decomposition output", () => {
    expect(isErrorOutput(DECOMPOSITION)).toBe(false);
  });

  it("returns true for message-only error payload (no 'error' key)", () => {
    const messageOnly: DecomposeErrorOutput = {
      type: "error",
      message: "Something failed",
    };
    expect(isErrorOutput(messageOnly as unknown as DecomposeGoalOutput)).toBe(
      true,
    );
  });
});

// ---------------------------------------------------------------------------
// getDecomposeGoalOutput — output parsing
// ---------------------------------------------------------------------------

describe("getDecomposeGoalOutput", () => {
  it("parses a direct object output", () => {
    const part = { output: DECOMPOSITION };
    const result = getDecomposeGoalOutput(part);
    expect(result).not.toBeNull();
    expect(isDecompositionOutput(result!)).toBe(true);
  });

  it("parses a JSON-encoded string output", () => {
    const part = { output: JSON.stringify(DECOMPOSITION) };
    const result = getDecomposeGoalOutput(part);
    expect(result).not.toBeNull();
    expect(isDecompositionOutput(result!)).toBe(true);
    expect((result as TaskDecompositionOutput).goal).toBe(
      "Build a news summarizer",
    );
  });

  it("parses an error output object", () => {
    const part = { output: ERROR_OUTPUT };
    const result = getDecomposeGoalOutput(part);
    expect(result).not.toBeNull();
    expect(isErrorOutput(result!)).toBe(true);
  });

  it("returns null for falsy output", () => {
    expect(getDecomposeGoalOutput({ output: null })).toBeNull();
    expect(getDecomposeGoalOutput({ output: undefined })).toBeNull();
    expect(getDecomposeGoalOutput({ output: "" })).toBeNull();
  });

  it("returns null for a plain non-JSON string", () => {
    expect(getDecomposeGoalOutput({ output: "just text" })).toBeNull();
  });

  it("returns null for a non-object part", () => {
    expect(getDecomposeGoalOutput(null)).toBeNull();
    expect(getDecomposeGoalOutput("string")).toBeNull();
    expect(getDecomposeGoalOutput(42)).toBeNull();
  });

  it("returns null for an array-type output (not a valid shape)", () => {
    expect(
      getDecomposeGoalOutput({ output: ["not", "an", "object"] }),
    ).toBeNull();
  });

  it("classifies 'steps+goal' before 'error' when object has all three keys", () => {
    // Verify type discrimination precedence: steps+goal wins
    const mixed = { ...DECOMPOSITION, error: "some_error" };
    const part = { output: mixed };
    const result = getDecomposeGoalOutput(part);
    expect(result).not.toBeNull();
    expect(isDecompositionOutput(result!)).toBe(true);
  });

  it("returns message-only error when no error key but has message", () => {
    const messageOnly = { type: "error", message: "Something failed" };
    const result = getDecomposeGoalOutput({ output: messageOnly });
    expect(result).not.toBeNull();
    expect(isErrorOutput(result!)).toBe(true);
    expect((result as DecomposeErrorOutput).message).toBe("Something failed");
  });
});

// ---------------------------------------------------------------------------
// getAnimationText
// ---------------------------------------------------------------------------

describe("getAnimationText", () => {
  it("shows analyzing text during input-streaming", () => {
    const text = getAnimationText({ state: "input-streaming" });
    expect(text.toLowerCase()).toContain("analyzing");
  });

  it("shows analyzing text during input-available", () => {
    const text = getAnimationText({ state: "input-available" });
    expect(text.toLowerCase()).toContain("analyzing");
  });

  it("shows plan ready with step count on output-available with decomposition", () => {
    const text = getAnimationText({
      state: "output-available",
      output: DECOMPOSITION,
    });
    expect(text).toContain("3 steps");
  });

  it("shows analyzing when output-available but output is not a decomposition", () => {
    const text = getAnimationText({
      state: "output-available",
      output: null,
    });
    expect(text.toLowerCase()).toContain("analyzing");
  });

  it("shows error text on output-error state", () => {
    const text = getAnimationText({ state: "output-error" });
    expect(text.toLowerCase()).toContain("error");
  });

  it("falls back to analyzing for unknown state", () => {
    const text = getAnimationText({ state: "result" as never });
    expect(text.toLowerCase()).toContain("analyzing");
  });
});
