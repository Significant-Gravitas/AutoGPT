import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import { buildAnswersMessage, getPendingQuestions } from "../helpers";

type Message = UIMessage<unknown, UIDataTypes, UITools>;
type Part = Message["parts"][number];

function toolPart(overrides: Record<string, unknown>): Part {
  return {
    type: "tool-ask_question",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    output: {
      type: "agent_builder_clarification_needed",
      questions: [{ question: "Which region?", keyword: "region" }],
    },
    ...overrides,
  } as unknown as Part;
}

function assistantMessage(parts: Part[], id = "a1"): Message {
  return { id, role: "assistant", parts };
}

describe("getPendingQuestions", () => {
  it("returns null when there are no messages", () => {
    expect(getPendingQuestions([])).toBeNull();
  });

  it("returns null when a user reply supersedes the asking message", () => {
    expect(
      getPendingQuestions([
        assistantMessage([toolPart({})]),
        { id: "u1", role: "user", parts: [{ type: "text", text: "Europe" }] },
      ]),
    ).toBeNull();
  });

  it("returns null while the tool part has no output yet", () => {
    expect(
      getPendingQuestions([
        assistantMessage([
          toolPart({ state: "input-available", output: undefined }),
        ]),
      ]),
    ).toBeNull();
  });

  it("ignores non-tool parts and unrelated tool outputs", () => {
    expect(
      getPendingQuestions([
        assistantMessage([
          { type: "text", text: "Working on it" } as unknown as Part,
          toolPart({
            type: "tool-web_search",
            output: { type: "search_results" },
          }),
        ]),
      ]),
    ).toBeNull();
  });

  it("picks up clarification outputs from tools other than ask_question", () => {
    const pending = getPendingQuestions([
      assistantMessage([toolPart({ type: "tool-create_agent" })]),
    ]);

    expect(pending?.questions).toEqual([
      { question: "Which region?", keyword: "region" },
    ]);
  });

  it("parses JSON-string outputs and ignores unparseable ones", () => {
    const pending = getPendingQuestions([
      assistantMessage([
        toolPart({
          type: "tool-create_agent",
          output: JSON.stringify({
            type: "agent_builder_clarification_needed",
            questions: [{ question: "Which region?", keyword: "region" }],
          }),
        }),
      ]),
    ]);
    expect(pending?.questions).toHaveLength(1);

    expect(
      getPendingQuestions([
        assistantMessage([
          toolPart({ type: "tool-create_agent", output: "not json {" }),
        ]),
      ]),
    ).toBeNull();
  });

  it("returns null when every question is malformed", () => {
    expect(
      getPendingQuestions([
        assistantMessage([
          toolPart({
            output: {
              type: "agent_builder_clarification_needed",
              questions: [{ question: 123, keyword: "region" }],
            },
          }),
        ]),
      ]),
    ).toBeNull();
  });

  it("merges questions across parts into one dock entry", () => {
    const pending = getPendingQuestions([
      assistantMessage(
        [
          toolPart({ toolCallId: "call-1" }),
          toolPart({
            toolCallId: "call-2",
            output: {
              type: "agent_builder_clarification_needed",
              questions: [{ question: "Which format?", keyword: "format" }],
            },
          }),
        ],
        "msg-9",
      ),
    ]);

    expect(pending).toEqual({
      dockId: "msg-9:call-1+call-2",
      questions: [
        { question: "Which region?", keyword: "region" },
        { question: "Which format?", keyword: "format" },
      ],
      callIds: ["call-1", "call-2"],
    });
  });

  it("dedupes colliding keywords across merged parts", () => {
    const pending = getPendingQuestions([
      assistantMessage([
        toolPart({ toolCallId: "call-1" }),
        toolPart({
          toolCallId: "call-2",
          output: {
            type: "agent_builder_clarification_needed",
            questions: [{ question: "Which region again?", keyword: "region" }],
          },
        }),
      ]),
    ]);

    expect(pending?.questions.map((q) => q.keyword)).toEqual([
      "region",
      "region-1",
    ]);
  });
});

describe("buildAnswersMessage", () => {
  it("quotes each question above its trimmed answer", () => {
    const message = buildAnswersMessage(
      [
        { question: "Which region?", keyword: "region" },
        { question: "Which format?", keyword: "format" },
      ],
      { region: "  Europe  ", format: "CSV" },
    );

    expect(message).toBe(
      "**Here are my answers:**\n\n> Which region?\n\nEurope\n\n> Which format?\n\nCSV\n\nPlease proceed.",
    );
  });
});
