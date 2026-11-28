import { fireEvent, render, screen } from "@testing-library/react";
import { useState, type ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { QuestionAnswerField } from "../QuestionAnswerField";

type Question = ComponentProps<typeof QuestionAnswerField>["question"];

function renderField(value = "ねこ") {
  const onSubmit = vi.fn();
  render(
    <QuestionAnswerField
      question={{ question: "Which animal?", keyword: "animal" }}
      value={value}
      labelId="label-1"
      autoFocus={false}
      onChange={() => {}}
      onSubmit={onSubmit}
    />,
  );
  return { textarea: screen.getByRole("textbox"), onSubmit };
}

describe("QuestionAnswerField Enter handling", () => {
  it("advances the pager on a plain Enter", () => {
    const { textarea, onSubmit } = renderField();

    fireEvent.keyDown(textarea, { key: "Enter" });

    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("does not advance while an IME owns the Enter", () => {
    const { textarea, onSubmit } = renderField();

    fireEvent.keyDown(textarea, { key: "Enter", isComposing: true });
    // Safari's confirming Enter arrives after compositionend.
    fireEvent.keyDown(textarea, { key: "Enter", keyCode: 229 });

    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("leaves Shift+Enter as the newline", () => {
    const { textarea, onSubmit } = renderField();

    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });

    expect(onSubmit).not.toHaveBeenCalled();
  });
});

function ControlledField({ question }: { question: Question }) {
  const [value, setValue] = useState("");
  return (
    <QuestionAnswerField
      question={question}
      value={value}
      labelId="label-1"
      autoFocus={false}
      onChange={setValue}
      onSubmit={() => {}}
    />
  );
}

describe("QuestionAnswerField Use example", () => {
  const withExample: Question = {
    question: "Which subject line?",
    keyword: "subject",
    example: "Weekly sales summary",
  };

  it("fills the empty answer with the example and focuses it", () => {
    render(<ControlledField question={withExample} />);
    const textarea = screen.getByRole("textbox") as HTMLTextAreaElement;

    fireEvent.click(
      screen.getByRole("button", {
        name: "Use example: Weekly sales summary",
      }),
    );

    expect(textarea.value).toBe("Weekly sales summary");
    expect(document.activeElement).toBe(textarea);
    expect(screen.queryByRole("button", { name: /^Use example/ })).toBeNull();
  });

  it("hides once the user has typed and comes back when the answer is cleared", () => {
    render(<ControlledField question={withExample} />);
    const textarea = screen.getByRole("textbox");

    fireEvent.change(textarea, { target: { value: "Q3 numbers" } });
    expect(screen.queryByRole("button", { name: /^Use example/ })).toBeNull();
    expect(screen.getByRole("textbox")).toBe(textarea);

    fireEvent.change(textarea, { target: { value: "" } });
    expect(screen.getByRole("button", { name: /^Use example/ })).toBeDefined();
    expect(screen.getByRole("textbox")).toBe(textarea);
  });

  it("is not offered without an example", () => {
    render(
      <ControlledField
        question={{ question: "Which subject line?", keyword: "subject" }}
      />,
    );

    expect(screen.queryByRole("button", { name: /^Use example/ })).toBeNull();
  });

  it("is not offered for a question with options", () => {
    render(
      <ControlledField
        question={{
          question: "Which region?",
          keyword: "region",
          options: ["Europe", "Asia"],
          example: "Europe, Asia",
        }}
      />,
    );

    expect(screen.queryByRole("button", { name: /^Use example/ })).toBeNull();
  });
});
