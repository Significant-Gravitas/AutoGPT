import { render } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { memo, useContext } from "react";
import { describe, expect, it } from "vitest";
import { PendingOnboardingContext } from "../../../ExpertOnboardingCard/PendingOnboardingContext";
import { AnsweredQuestionsContext } from "../../../QuestionDock/AnsweredQuestionsContext";
import { PendingQuestionsContext } from "../../../QuestionDock/PendingQuestionsContext";
import { CurrentPauseContext } from "../../../TurnPausedCard/CurrentPauseContext";
import { HeldOutcomesContext } from "../../HeldOutcomesContext";
import { PendingAnswerContexts } from "../PendingAnswerContexts";

type Message = UIMessage<unknown, UIDataTypes, UITools>;
type Part = Message["parts"][number];

interface Seen {
  questions: unknown;
  answered: ReadonlyMap<string, ReadonlyMap<string, string>>;
  onboarding: unknown;
  held: ReadonlyMap<string, unknown>;
  pause: unknown;
}

function heldResult(toolCallId: string, body: string): Message {
  return {
    id: `held-${toolCallId}`,
    role: "user",
    metadata: { held_call: { tool_call_id: toolCallId, outcome: "approved" } },
    parts: [
      {
        type: "text",
        text: `<held_call_result tool="create_folder" tool_call_id="${toolCallId}" review_id="r">\n${body}\n</held_call_result>`,
      },
    ],
  };
}

const askPart = {
  type: "tool-ask_question",
  toolCallId: "ask-1",
  state: "output-available",
  input: {},
  output: {
    type: "agent_builder_clarification_needed",
    questions: [
      {
        question: "Which region?",
        keyword: "region",
        options: ["Europe", "Asia"],
        allow_multiple: true,
      },
    ],
  },
} as unknown as Part;

function assistant(text: string, extra: Part[] = []): Message {
  return {
    id: "a1",
    role: "assistant",
    parts: [askPart, ...extra, { type: "text", text }],
  };
}

function setup(initial: Message[]) {
  const seen: Seen[] = [];
  const Consumer = memo(function Consumer() {
    seen.push({
      questions: useContext(PendingQuestionsContext),
      answered: useContext(AnsweredQuestionsContext),
      onboarding: useContext(PendingOnboardingContext),
      held: useContext(HeldOutcomesContext),
      pause: useContext(CurrentPauseContext),
    });
    return null;
  });
  const consumer = <Consumer />;
  const view = render(
    <PendingAnswerContexts messages={initial}>
      {consumer}
    </PendingAnswerContexts>,
  );
  function rerender(messages: Message[]) {
    view.rerender(
      <PendingAnswerContexts messages={messages}>
        {consumer}
      </PendingAnswerContexts>,
    );
  }
  return { seen, rerender };
}

describe("PendingAnswerContexts", () => {
  it("publishes identical values when rendered again with the same messages", () => {
    const messages = [heldResult("call-7", '{"ok":true}'), assistant("Hi")];
    const { seen, rerender } = setup(messages);
    const first = seen[0];
    expect(first.questions).not.toBeNull();
    expect(first.held.size).toBe(1);

    rerender(messages);

    // The memoised consumer did not render again: no value changed.
    expect(seen).toHaveLength(1);
  });

  it("keeps every value's identity while a token streams into the last message", () => {
    const held = heldResult("call-7", '{"ok":true}');
    const { seen, rerender } = setup([held, assistant("Hel")]);

    rerender([held, assistant("Hello")]);
    rerender([held, assistant("Hello there")]);

    expect(seen).toHaveLength(1);
    expect(seen[0].held.get("call-7")).toEqual({
      outcome: "approved",
      output: { ok: true },
    });
  });

  it("keeps the held outcomes when the result message is rebuilt with the same content", () => {
    const { seen, rerender } = setup([
      heldResult("call-7", '{"ok":true}'),
      assistant("Hi"),
    ]);

    rerender([heldResult("call-7", '{"ok":true}'), assistant("Hi")]);

    expect(seen).toHaveLength(1);
  });

  it("publishes a new map once a held call is answered", () => {
    const first = heldResult("call-7", '{"ok":true}');
    const { seen, rerender } = setup([first, assistant("Hi")]);

    rerender([first, heldResult("call-8", "posted"), assistant("Hi")]);

    expect(seen).toHaveLength(2);
    expect(seen[1].held).not.toBe(seen[0].held);
    expect([...seen[1].held.keys()]).toEqual(["call-7", "call-8"]);
    expect(seen[1].questions).toBe(seen[0].questions);
  });

  it("publishes new questions when the asking call changes", () => {
    const { seen, rerender } = setup([assistant("Hi")]);
    const otherAsk = {
      ...askPart,
      toolCallId: "ask-2",
      output: {
        type: "agent_builder_clarification_needed",
        questions: [{ question: "Which team?", keyword: "team" }],
      },
    } as unknown as Part;

    rerender([assistant("Hi", [otherAsk])]);

    expect(seen).toHaveLength(2);
    expect(seen[1].questions).toMatchObject({
      callIds: ["ask-1", "ask-2"],
    });
  });

  it("clears the questions once the user replies", () => {
    const { seen, rerender } = setup([assistant("Hi")]);

    rerender([
      assistant("Hi"),
      { id: "u1", role: "user", parts: [{ type: "text", text: "Europe" }] },
    ]);

    expect(seen).toHaveLength(2);
    expect(seen[1].questions).toBeNull();
  });

  it("reads the answers back from the user's reply", () => {
    const { seen, rerender } = setup([assistant("Hi")]);

    rerender([
      assistant("Hi"),
      {
        id: "u1",
        role: "user",
        parts: [
          {
            type: "text",
            text: "**Here are my answers:**\n\n> Which region?\n\n- Europe\n- Asia\n\nPlease proceed.",
          },
        ],
      },
    ]);

    expect(seen[1].answered.get("ask-1")?.get("region")).toBe("Europe, Asia");
  });

  it("shares one empty map across different messages when nothing is held", () => {
    const { seen, rerender } = setup([
      { id: "u1", role: "user", parts: [{ type: "text", text: "Hi" }] },
    ]);
    expect(seen[0].held.size).toBe(0);

    rerender([
      { id: "u1", role: "user", parts: [{ type: "text", text: "Hi" }] },
      { id: "a1", role: "assistant", parts: [{ type: "text", text: "Yo" }] },
    ]);

    expect(seen).toHaveLength(1);
  });

  it("passes the paused part's own data object through by reference", () => {
    const data = { reason: "daily", resets_at: null };
    const paused = {
      id: "a1",
      role: "assistant",
      parts: [{ type: "data-turn-paused", data }],
    } as unknown as Message;
    const { seen, rerender } = setup([paused]);
    expect(seen[0].pause).toBe(data);

    const equalData = { reason: "daily", resets_at: null };
    rerender([
      {
        id: "a1",
        role: "assistant",
        parts: [{ type: "data-turn-paused", data: equalData }],
      } as unknown as Message,
    ]);

    // Cards match their pause by reference, so an equal copy is a new pause.
    expect(seen).toHaveLength(2);
    expect(seen[1].pause).toBe(equalData);
  });
});
