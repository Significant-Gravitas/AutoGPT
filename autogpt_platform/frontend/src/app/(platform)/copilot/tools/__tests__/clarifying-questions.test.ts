import { describe, expect, it } from "vitest";
import {
  buildClarificationAnswersMessage,
  extractClarifyingQuestions,
  formatAnswer,
  isAnswered,
  normalizeClarifyingQuestions,
  toAnswerList,
  toMultiAnswer,
} from "../clarifying-questions";

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

  it("keeps string options and drops blank or non-string entries", () => {
    const result = normalizeClarifyingQuestions([
      { question: "Q1", keyword: "k1", options: ["Email", "  ", 42, "Slack"] },
      { question: "Q2", keyword: "k2", options: [] },
      { question: "Q3", keyword: "k3", options: "Email" },
    ]);
    expect(result[0].options).toEqual(["Email", "Slack"]);
    expect(result[1].options).toBeUndefined();
    expect(result[2].options).toBeUndefined();
  });

  it("trims padded options and drops duplicates", () => {
    const result = normalizeClarifyingQuestions([
      { question: "Q1", keyword: "k1", options: [" Email ", "Slack", "Email"] },
    ]);
    expect(result[0].options).toEqual(["Email", "Slack"]);
  });
});

describe("extractClarifyingQuestions", () => {
  it("carries the multi-select flag through from the output", () => {
    const result = extractClarifyingQuestions({
      output: {
        questions: [
          {
            question: "Which areas?",
            keyword: "areas",
            options: ["Research", "Outreach"],
            allow_multiple: true,
          },
        ],
      },
    });
    expect(result[0].allow_multiple).toBe(true);
  });

  it("recovers the multi-select flag from the input the model sent", () => {
    const result = extractClarifyingQuestions({
      input: {
        questions: [
          {
            question: "Which areas?",
            keyword: "areas",
            options: ["Research", "Outreach"],
            allow_multiple: true,
          },
        ],
      },
      output: {
        questions: [
          {
            question: "Which areas?",
            keyword: "areas",
            example: "Research, Outreach",
          },
        ],
      },
    });
    expect(result[0].allow_multiple).toBe(true);
    expect(result[0].options).toEqual(["Research", "Outreach"]);
  });

  it("leaves a question single-select when nothing asked for more", () => {
    const result = extractClarifyingQuestions({
      output: {
        questions: [
          {
            question: "Which channel?",
            keyword: "channel",
            options: ["Email", "Slack"],
          },
        ],
      },
    });
    expect(result[0].allow_multiple).toBeUndefined();
  });

  it("drops the multi-select flag from a question with no options", () => {
    const result = extractClarifyingQuestions({
      output: {
        questions: [
          {
            question: "Anything else?",
            keyword: "notes",
            allow_multiple: true,
          },
        ],
      },
    });
    expect(result[0].allow_multiple).toBeUndefined();
  });

  it("reads options straight from the output when present", () => {
    const result = extractClarifyingQuestions({
      output: {
        questions: [
          {
            question: "Which channel?",
            keyword: "channel",
            options: ["Email", "Slack"],
          },
        ],
      },
    });
    expect(result[0].options).toEqual(["Email", "Slack"]);
  });

  it("recovers options from the input when the output collapsed them", () => {
    const result = extractClarifyingQuestions({
      input: {
        questions: [
          {
            question: "Which channel?",
            keyword: "channel",
            options: ["Email", "Slack"],
          },
        ],
      },
      output: {
        questions: [
          {
            question: "Which channel?",
            keyword: "channel",
            example: "Email, Slack",
          },
        ],
      },
    });
    expect(result[0].options).toEqual(["Email", "Slack"]);
    expect(result[0].example).toBe("Email, Slack");
  });

  it("keeps same-worded questions apart when recovering from the input", () => {
    const result = extractClarifyingQuestions({
      input: {
        questions: [
          {
            question: "Which channel?",
            keyword: "source",
            options: ["Email", "Slack"],
          },
          {
            question: "Which channel?",
            keyword: "destination",
            options: ["Notion", "Drive"],
          },
        ],
      },
      output: {
        questions: [
          { question: "Which channel?", keyword: "source" },
          { question: "Which channel?", keyword: "destination" },
        ],
      },
    });
    expect(result[0].options).toEqual(["Email", "Slack"]);
    expect(result[1].options).toEqual(["Notion", "Drive"]);
  });

  it("keeps a keyword/question pair that a space separator would collide", () => {
    // "a" + "b c" and "a b" + "c" join to the same string under a space, so
    // the recovery key has to use a separator neither half can contain.
    const result = extractClarifyingQuestions({
      input: {
        questions: [
          { question: "b c", keyword: "a", options: ["Email", "Slack"] },
          { question: "c", keyword: "a b", options: ["Notion", "Drive"] },
        ],
      },
      output: {
        questions: [
          { question: "b c", keyword: "a" },
          { question: "c", keyword: "a b" },
        ],
      },
    });
    expect(result[0].options).toEqual(["Email", "Slack"]);
    expect(result[1].options).toEqual(["Notion", "Drive"]);
  });

  it("falls back to the input while the call is still in flight", () => {
    const result = extractClarifyingQuestions({
      input: {
        questions: [
          {
            question: "Which channel?",
            keyword: "channel",
            options: ["Email", "Slack"],
          },
        ],
      },
    });
    expect(result).toHaveLength(1);
    expect(result[0].question).toBe("Which channel?");
    expect(result[0].options).toEqual(["Email", "Slack"]);
  });

  it("leaves options unset when neither side carries them", () => {
    const result = extractClarifyingQuestions({
      output: {
        questions: [{ question: "Which channel?", keyword: "channel" }],
      },
    });
    expect(result[0].options).toBeUndefined();
  });
});

describe("multi-select answers", () => {
  it("reads one answer and many through the same list", () => {
    expect(toAnswerList("  Europe  ")).toEqual(["Europe"]);
    expect(
      toAnswerList({ selected: ["Research"], custom: " Outreach " }),
    ).toEqual(["Research", "Outreach"]);
    expect(toAnswerList({ selected: ["Research"], custom: "  " })).toEqual([
      "Research",
    ]);
  });

  it("keeps typed text after the ticks, even when it equals one", () => {
    expect(
      toAnswerList({ selected: ["Research"], custom: "Research" }),
    ).toEqual(["Research", "Research"]);
  });

  it("treats a blank answer of either shape as unanswered", () => {
    expect(isAnswered(undefined)).toBe(false);
    expect(isAnswered("   ")).toBe(false);
    expect(isAnswered({ selected: [], custom: "" })).toBe(false);
    expect(isAnswered({ selected: [], custom: " " })).toBe(false);
    expect(isAnswered({ selected: ["Research"], custom: "" })).toBe(true);
    expect(isAnswered({ selected: [], custom: "Partnerships" })).toBe(true);
  });

  it("bullets several picks and leaves one inline", () => {
    expect(
      formatAnswer({ selected: ["Research", "Outreach"], custom: "" }),
    ).toBe("- Research\n- Outreach");
    expect(formatAnswer({ selected: ["Research"], custom: "" })).toBe(
      "Research",
    );
    expect(formatAnswer("Europe")).toBe("Europe");
  });

  it("reads a multi-select field's value from either answer shape", () => {
    expect(toMultiAnswer(undefined)).toEqual({ selected: [], custom: "" });
    expect(toMultiAnswer("Partnerships")).toEqual({
      selected: [],
      custom: "Partnerships",
    });
    const answer = { selected: ["Research"], custom: "Research" };
    expect(toMultiAnswer(answer)).toBe(answer);
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

  it("lists a multi-select answer under its question", () => {
    const result = buildClarificationAnswersMessage(
      { areas: { selected: ["Research", "Outreach"], custom: "" } },
      [{ question: "Which areas?", keyword: "areas" }],
      "create",
    );
    expect(result).toContain("> Which areas?\n\n- Research\n- Outreach");
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
