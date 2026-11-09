import { describe, expect, it } from "vitest";
import {
  extractClarifyingQuestions,
  initialAnswers,
} from "../clarifying-questions";

describe("initialAnswers", () => {
  it("starts every question that has a guess at that guess", () => {
    expect(
      initialAnswers([
        { question: "Which team?", keyword: "team", default_guess: "Sales" },
        { question: "When?", keyword: "when" },
      ]),
    ).toEqual({ team: "Sales" });
  });

  it("ticks a guess that is one of a multi-select's options", () => {
    expect(
      initialAnswers([
        {
          question: "Which roles?",
          keyword: "roles",
          options: ["Ops", "Sales"],
          allow_multiple: true,
          default_guess: "Ops",
        },
        {
          question: "Which tools?",
          keyword: "tools",
          options: ["Slack"],
          allow_multiple: true,
          default_guess: "Teams",
        },
      ]),
    ).toEqual({
      roles: { selected: ["Ops"], custom: "" },
      tools: { selected: [], custom: "Teams" },
    });
  });
});

describe("extractClarifyingQuestions", () => {
  it("keeps the guess the tool returned, trimmed", () => {
    const [question] = extractClarifyingQuestions({
      output: {
        questions: [
          {
            question: "Which team?",
            keyword: "team",
            default_guess: " Sales ",
          },
        ],
      },
    });
    expect(question.default_guess).toBe("Sales");
  });
});
