import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { ChainRow } from "../helpers";
import { QuestionRowForm } from "../QuestionRowForm";

function questionRow(questions: unknown[]): ChainRow {
  return {
    key: "question",
    category: "question",
    text: "Asked a question",
    state: "done",
    output: { questions },
  };
}

describe("QuestionRowForm", () => {
  afterEach(cleanup);

  it("ignores malformed questions and renders valid questions", () => {
    render(
      <CopilotChatActionsProvider onSend={vi.fn()}>
        <QuestionRowForm
          row={questionRow([
            { question: 123, keyword: "invalid" },
            { question: "Which region?", keyword: "region" },
          ])}
        />
      </CopilotChatActionsProvider>,
    );

    expect(screen.getByLabelText("Which region?")).toBeDefined();
    expect(screen.queryByText("123")).toBeNull();
  });

  it("focuses the first missing answer and only sends complete answers", async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(
      <CopilotChatActionsProvider onSend={onSend}>
        <QuestionRowForm
          row={questionRow([
            { question: "Which region?", keyword: "region" },
            { question: "Which format?", keyword: "format" },
          ])}
        />
      </CopilotChatActionsProvider>,
    );

    await user.click(screen.getByRole("button", { name: "Proceed" }));
    expect(screen.getByLabelText("Which region?")).toBe(document.activeElement);
    expect(onSend).not.toHaveBeenCalled();

    await user.type(screen.getByLabelText("Which region?"), "Europe");
    await user.type(screen.getByLabelText("Which format?"), "CSV");
    await user.keyboard("{Enter}");

    expect(onSend).toHaveBeenCalledOnce();
    expect(onSend).toHaveBeenCalledWith(
      "**Here are my answers:**\n\n> Which region?\n\nEurope\n\n> Which format?\n\nCSV\n\nPlease proceed.",
    );
    expect(screen.getByText("Europe")).toBeDefined();
  });

  it("falls back to a read-only question card without chat actions", () => {
    render(
      <QuestionRowForm
        row={questionRow([
          { question: "Which region?", keyword: "region", example: "Europe" },
        ])}
      />,
    );

    expect(screen.getByText("Which region?")).toBeDefined();
    expect(screen.getByText("e.g. Europe")).toBeDefined();
    expect(screen.queryByRole("textbox")).toBeNull();
  });
});
