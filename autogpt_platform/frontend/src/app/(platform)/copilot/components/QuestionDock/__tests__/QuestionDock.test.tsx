import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { QuestionDock } from "../QuestionDock";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

function questionMessage(questions: unknown[], id = "a1"): Message {
  return {
    id,
    role: "assistant",
    parts: [
      {
        type: "tool-ask_question",
        toolCallId: `${id}-call`,
        state: "output-available",
        input: {},
        output: { type: "agent_builder_clarification_needed", questions },
      } as Message["parts"][number],
    ],
  };
}

describe("QuestionDock", () => {
  afterEach(cleanup);

  function renderDock(messages: Message[], onSend = vi.fn()) {
    render(
      <CopilotChatActionsProvider onSend={onSend}>
        <QuestionDock messages={messages} />
      </CopilotChatActionsProvider>,
    );
    return onSend;
  }

  it("renders nothing when the last message is from the user", () => {
    renderDock([
      questionMessage([{ question: "Which region?", keyword: "region" }]),
      { id: "u1", role: "user", parts: [{ type: "text", text: "Europe" }] },
    ]);

    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("ignores malformed questions and renders valid questions", () => {
    renderDock([
      questionMessage([
        { question: 123, keyword: "invalid" },
        { question: "Which region?", keyword: "region", example: "Europe" },
      ]),
    ]);

    expect(screen.getByLabelText("Which region?")).toBeDefined();
    expect(screen.getByPlaceholderText("e.g. Europe")).toBeDefined();
    expect(screen.queryByText("123")).toBeNull();
  });

  it("focuses the first missing answer and only sends complete answers", async () => {
    const user = userEvent.setup();
    const onSend = renderDock([
      questionMessage([
        { question: "Which region?", keyword: "region" },
        { question: "Which format?", keyword: "format" },
      ]),
    ]);

    await user.click(screen.getByRole("button", { name: "Answer" }));
    expect(screen.getByLabelText("Which region?")).toBe(document.activeElement);
    expect(onSend).not.toHaveBeenCalled();

    await user.type(screen.getByLabelText("Which region?"), "Europe");
    await user.type(screen.getByLabelText("Which format?"), "CSV");
    await user.keyboard("{Enter}");

    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith(
      "**Here are my answers:**\n\n> Which region?\n\nEurope\n\n> Which format?\n\nCSV\n\nPlease proceed.",
    );
    expect(screen.queryByRole("textbox")).toBeNull();
  });

  it("keeps a multi-select answer as a list and sends the picks as bullets", async () => {
    const user = userEvent.setup();
    const onSend = renderDock([
      questionMessage([
        {
          question: "What areas should they own?",
          keyword: "areas",
          options: ["Research", "Outreach", "Reporting"],
          allow_multiple: true,
        },
      ]),
    ]);

    expect(
      screen.getByRole("group", { name: "What areas should they own?" }),
    ).toBeDefined();
    await user.click(screen.getByRole("checkbox", { name: "Research" }));
    await user.click(screen.getByRole("checkbox", { name: "Reporting" }));
    expect(
      screen
        .getByRole("checkbox", { name: "Outreach" })
        .getAttribute("aria-checked"),
    ).toBe("false");

    await user.click(screen.getByRole("button", { name: "Answer" }));

    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith(
      "**Here are my answers:**\n\n> What areas should they own?\n\n- Research\n- Reporting\n\nPlease proceed.",
    );
  });

  it("focuses an unanswered multi-select question instead of sending", async () => {
    const user = userEvent.setup();
    const onSend = renderDock([
      questionMessage([
        { question: "Which region?", keyword: "region" },
        {
          question: "What areas should they own?",
          keyword: "areas",
          options: ["Research", "Outreach", "Reporting"],
          allow_multiple: true,
        },
      ]),
    ]);

    await user.type(screen.getByLabelText("Which region?"), "Europe");
    await user.click(screen.getByRole("button", { name: "Answer" }));

    expect(onSend).not.toHaveBeenCalled();
    expect(document.activeElement).toBe(
      screen.getByRole("checkbox", { name: "Research" }),
    );

    await user.keyboard("{ArrowDown}{Enter}");
    await user.click(screen.getByRole("button", { name: "Answer" }));
    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith(
      "**Here are my answers:**\n\n> Which region?\n\nEurope\n\n> What areas should they own?\n\nOutreach\n\nPlease proceed.",
    );
  });

  it("hides after skipping", async () => {
    const user = userEvent.setup();
    const onSend = renderDock([
      questionMessage([{ question: "Which region?", keyword: "region" }]),
    ]);

    await user.click(screen.getByRole("button", { name: "Skip" }));

    expect(screen.queryByRole("textbox")).toBeNull();
    expect(onSend).not.toHaveBeenCalled();
  });

  it("renders nothing without chat actions", () => {
    render(
      <QuestionDock
        messages={[
          questionMessage([{ question: "Which region?", keyword: "region" }]),
        ]}
      />,
    );

    expect(screen.queryByRole("textbox")).toBeNull();
  });
});
