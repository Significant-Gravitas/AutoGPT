import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import type { ClarifyingQuestion } from "../../../tools/clarifying-questions";
import { buildAnswersMessage } from "../helpers";
import { getAnsweredQuestions } from "../answeredQuestions";
import { buildOnboardingAnswersMessage } from "../../ExpertOnboardingCard/helpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;
type Part = Message["parts"][number];

function ask(callId: string, questions: ClarifyingQuestion[]): Part {
  return {
    type: "tool-ask_question",
    toolCallId: callId,
    state: "output-available",
    input: {},
    output: { type: "agent_builder_clarification_needed", questions },
  } as Part;
}

function assistant(parts: Part[], id = "a1"): Message {
  return { id, role: "assistant", parts };
}

function reply(text: string, id = "u1"): Message {
  return { id, role: "user", parts: [{ type: "text", text }] };
}

const region = { question: "Which region?", keyword: "region" };

describe("getAnsweredQuestions", () => {
  it("keeps separate answers to identical prompts within one call", () => {
    const questions = [
      { question: "Which channel?", keyword: "source" },
      { question: "Which channel?", keyword: "destination" },
    ];
    const answers = getAnsweredQuestions([
      assistant([ask("call-1", questions)]),
      reply(
        buildAnswersMessage(questions, {
          source: "Email",
          destination: "Drive",
        }),
      ),
    ]);
    expect(answers.get("call-1")).toEqual(
      new Map([
        ["source", "Email"],
        ["destination", "Drive"],
      ]),
    );
  });

  it("keeps separate answers to identical prompts across calls", () => {
    const questions = [{ question: "Which channel?", keyword: "channel" }];
    const answers = getAnsweredQuestions([
      assistant([ask("source", questions), ask("destination", questions)]),
      reply(
        buildAnswersMessage(
          [...questions, { ...questions[0], keyword: "channel-1" }],
          { channel: "Email", "channel-1": "Drive" },
        ),
      ),
    ]);
    expect(answers.get("source")?.get("channel")).toBe("Email");
    expect(answers.get("destination")?.get("channel")).toBe("Drive");
  });

  it("reads onboarding answers by step keyword", () => {
    const steps = [
      {
        question: "Which channel?",
        keyword: "source",
        options: [],
        allow_multiple: false,
      },
      {
        question: "Which channel?",
        keyword: "destination",
        options: [],
        allow_multiple: false,
      },
    ];
    const part = {
      type: "tool-expert_onboarding",
      toolCallId: "hire",
      state: "output-available",
      input: {},
      output: { type: "expert_onboarding", steps },
    } as Part;
    const answers = getAnsweredQuestions([
      assistant([part]),
      reply(
        buildOnboardingAnswersMessage(steps, {
          source: "Email",
          destination: "Drive",
        }),
      ),
    ]);
    expect(answers.get("hire")).toEqual(
      new Map([
        ["source", "Email"],
        ["destination", "Drive"],
      ]),
    );
  });

  it("preserves multiline questions and typed answers", () => {
    const questions = [
      {
        question: "Which region?\n\nInclude any exclusions.",
        keyword: "region",
      },
    ];
    const answer =
      "Europe\n\n> Budget note\n\n- Exclude France\n- Include Spain";
    const answers = getAnsweredQuestions([
      assistant([ask("call-1", questions)]),
      reply(buildAnswersMessage(questions, { region: answer })),
    ]);
    expect(answers.get("call-1")?.get("region")).toBe(answer);
  });

  it("does not mistake typed 'Please proceed' text for the footer", () => {
    const answer = "Europe\n\nPlease proceed only after checking the budget.";
    const answers = getAnsweredQuestions([
      assistant([ask("call-1", [region])]),
      reply(buildAnswersMessage([region], { region: answer })),
    ]);
    expect(answers.get("call-1")?.get("region")).toBe(answer);
  });

  it("shows multi-select picks on one line", () => {
    const questions = [
      { ...region, options: ["Europe", "Asia"], allow_multiple: true },
    ];
    const answers = getAnsweredQuestions([
      assistant([ask("call-1", questions)]),
      reply(
        buildAnswersMessage(questions, {
          region: { selected: ["Europe", "Asia"], custom: "" },
        }),
      ),
    ]);
    expect(answers.get("call-1")?.get("region")).toBe("Europe, Asia");
  });

  it("keeps typed bullet lists in single-answer questions", () => {
    const answer = "- Europe\n- Asia";
    const answers = getAnsweredQuestions([
      assistant([ask("call-1", [region])]),
      reply(buildAnswersMessage([region], { region: answer })),
    ]);
    expect(answers.get("call-1")?.get("region")).toBe(answer);
  });

  it("reads all answer sections in a combined chain reply", () => {
    const format = { question: "Which format?", keyword: "format" };
    const answers = getAnsweredQuestions([
      assistant([ask("region", [region]), ask("format", [format])]),
      reply(
        [
          buildAnswersMessage([format], { format: "CSV" }),
          buildAnswersMessage([region], { region: "Europe" }),
        ].join("\n\n"),
      ),
    ]);
    expect(answers.get("region")?.get("region")).toBe("Europe");
    expect(answers.get("format")?.get("format")).toBe("CSV");
  });

  it("uses the first subsequent user reply and does not fill skipped questions later", () => {
    const answers = getAnsweredQuestions([
      assistant([ask("skipped", [region])]),
      reply("Skip questions"),
      assistant([ask("answered", [region])], "a2"),
      reply(buildAnswersMessage([region], { region: "Europe" }), "u2"),
    ]);
    expect(answers.has("skipped")).toBe(false);
    expect(answers.get("answered")?.get("region")).toBe("Europe");
  });

  it("ignores unanswered calls, unrelated tools and unstructured replies", () => {
    expect(getAnsweredQuestions([]).size).toBe(0);
    expect(
      getAnsweredQuestions([assistant([ask("call-1", [region])])]).size,
    ).toBe(0);
    expect(
      getAnsweredQuestions([
        assistant([ask("call-1", [region])]),
        reply("Europe"),
      ]).size,
    ).toBe(0);
    const part = {
      type: "tool-web_search",
      toolCallId: "search",
      state: "output-available",
      input: {},
      output: {},
    } as Part;
    expect(
      getAnsweredQuestions([
        assistant([part]),
        reply(buildAnswersMessage([region], { region: "Europe" })),
      ]).size,
    ).toBe(0);
  });
});
