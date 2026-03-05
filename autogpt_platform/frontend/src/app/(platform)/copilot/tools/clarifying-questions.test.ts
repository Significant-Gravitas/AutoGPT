import { describe, expect, it } from "vitest";
import {
  buildClarificationAnswersMessage,
  normalizeClarifyingQuestions,
} from "./clarifying-questions";

describe("normalizeClarifyingQuestions", () => {
  it("returns normalized questions with trimmed lowercase keywords", () => {
    const result = normalizeClarifyingQuestions([
      { question: "What is your goal?", keyword: "  Goal  ", example: "test" },
    ]);
    expect(result).toEqual([
      { question: "What is your goal?", keyword: "goal", example: "test" },
    ]);
  });

  it("deduplicates keywords by appending a numeric suffix", () => {
    const result = normalizeClarifyingQuestions([
      { question: "Q1", keyword: "topic" },
      { question: "Q2", keyword: "topic" },
      { question: "Q3", keyword: "topic" },
    ]);
    expect(result.map((q) => q.keyword)).toEqual([
      "topic",
      "topic-1",
      "topic-2",
    ]);
  });

  it("falls back to question-{index} when keyword is empty", () => {
    const result = normalizeClarifyingQuestions([
      { question: "First?", keyword: "" },
      { question: "Second?", keyword: "  " },
    ]);
    expect(result[0].keyword).toBe("question-0");
    expect(result[1].keyword).toBe("question-1");
  });

  it("coerces non-string examples to undefined", () => {
    const result = normalizeClarifyingQuestions([
      { question: "Q1", keyword: "k1", example: 42 },
      { question: "Q2", keyword: "k2", example: null },
      { question: "Q3", keyword: "k3", example: { nested: true } },
    ]);
    expect(result[0].example).toBeUndefined();
    expect(result[1].example).toBeUndefined();
    expect(result[2].example).toBeUndefined();
  });

  it("trims string examples and omits empty ones", () => {
    const result = normalizeClarifyingQuestions([
      { question: "Q1", keyword: "k1", example: "  valid  " },
      { question: "Q2", keyword: "k2", example: "   " },
    ]);
    expect(result[0].example).toBe("valid");
    expect(result[1].example).toBeUndefined();
  });

  it("returns an empty array for empty input", () => {
    expect(normalizeClarifyingQuestions([])).toEqual([]);
  });
});

describe("buildClarificationAnswersMessage", () => {
  it("formats answers with create mode", () => {
    const result = buildClarificationAnswersMessage(
      { goal: "automate tasks" },
      [{ question: "What is your goal?", keyword: "goal" }],
      "create",
    );
    expect(result).toContain("> What is your goal?");
    expect(result).toContain("automate tasks");
    expect(result).toContain("Please proceed with creating the agent.");
  });

  it("formats answers with edit mode", () => {
    const result = buildClarificationAnswersMessage(
      { goal: "fix bugs" },
      [{ question: "What should change?", keyword: "goal" }],
      "edit",
    );
    expect(result).toContain("Please proceed with editing the agent.");
  });

  it("uses empty string for missing answers", () => {
    const result = buildClarificationAnswersMessage(
      {},
      [{ question: "Q?", keyword: "missing" }],
      "create",
    );
    expect(result).toContain("> Q?\n\n");
  });
});
