import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, describe, expect, it } from "vitest";
import type {
  MultiAnswer,
  QuestionAnswer,
} from "../../../tools/clarifying-questions";
import { QuestionMultiAnswerField } from "../QuestionMultiAnswerField";
import { QuestionsSection } from "../QuestionsSection";

const options = ["Research", "Outreach", "Reporting"];

// The field is controlled, so every keystroke round-trips through state the
// way it does under the chain card; a fired change event would not catch text
// that is lost between one render and the next.
function FieldHarness({ initial }: { initial: MultiAnswer }) {
  const [value, setValue] = useState<MultiAnswer>(initial);
  return (
    <>
      <span id="areas">What areas should they own?</span>
      <QuestionMultiAnswerField
        options={options}
        value={value}
        labelId="areas"
        autoFocus={false}
        onChange={setValue}
        onSubmit={() => undefined}
      />
      <output data-testid="answer">{JSON.stringify(value)}</output>
    </>
  );
}

function SectionHarness() {
  const [answers, setAnswers] = useState<Record<string, QuestionAnswer>>({});
  return (
    <>
      <QuestionsSection
        requests={[
          {
            id: "questions-1",
            questions: [
              {
                question: "What areas should they own?",
                keyword: "areas",
                options,
                allow_multiple: true,
              },
              { question: "Which region?", keyword: "region" },
            ],
            answers,
            onAnswer: (keyword, value) =>
              setAnswers((prev) => ({ ...prev, [keyword]: value })),
            onSkip: () => undefined,
          },
        ]}
        isReady={false}
        onProceed={() => undefined}
      />
      <output data-testid="answer">{JSON.stringify(answers.areas)}</output>
    </>
  );
}

function empty(): MultiAnswer {
  return { selected: [], custom: "" };
}

function answer(): unknown {
  return JSON.parse(screen.getByTestId("answer").textContent ?? "null");
}

function textbox(): HTMLTextAreaElement {
  return screen.getByRole("textbox") as HTMLTextAreaElement;
}

function ticked(): string[] {
  return screen
    .getAllByRole("checkbox")
    .filter((box) => box.getAttribute("aria-checked") === "true")
    .map((box) => box.textContent ?? "");
}

describe("QuestionMultiAnswerField", () => {
  afterEach(cleanup);

  it("keeps typed text whole when it passes through an option on the way", async () => {
    const user = userEvent.setup();
    render(<FieldHarness initial={empty()} />);

    await user.click(screen.getByRole("button", { name: "Type something…" }));
    await user.type(textbox(), "Research and development");

    expect(textbox().value).toBe("Research and development");
    expect(ticked()).toEqual([]);
    expect(answer()).toEqual({
      selected: [],
      custom: "Research and development",
    });
  });

  it("keeps text that exactly matches an option as the user's own words", async () => {
    const user = userEvent.setup();
    render(<FieldHarness initial={empty()} />);

    await user.click(screen.getByRole("button", { name: "Type something…" }));
    await user.type(textbox(), "Research");

    expect(textbox().value).toBe("Research");
    expect(ticked()).toEqual([]);
    expect(answer()).toEqual({ selected: [], custom: "Research" });
  });

  it("lets an existing custom answer be edited through an option value", async () => {
    const user = userEvent.setup();
    render(<FieldHarness initial={{ selected: [], custom: "Researcher" }} />);

    expect(textbox().value).toBe("Researcher");
    await user.type(textbox(), "{backspace}{backspace}");
    expect(textbox().value).toBe("Research");
    expect(ticked()).toEqual([]);

    await user.type(textbox(), "ing");

    expect(textbox().value).toBe("Researching");
    expect(ticked()).toEqual([]);
    expect(answer()).toEqual({ selected: [], custom: "Researching" });
  });

  it("keeps the ticks and the text apart while both change", async () => {
    const user = userEvent.setup();
    render(<FieldHarness initial={{ selected: ["Research"], custom: "" }} />);

    await user.click(screen.getByRole("button", { name: "Type something…" }));
    await user.type(textbox(), "Research");
    expect(ticked()).toEqual(["Research"]);
    expect(textbox().value).toBe("Research");
    expect(answer()).toEqual({ selected: ["Research"], custom: "Research" });

    await user.click(screen.getByRole("checkbox", { name: "Outreach" }));
    expect(ticked()).toEqual(["Research", "Outreach"]);
    expect(textbox().value).toBe("Research");

    await user.click(screen.getByRole("checkbox", { name: "Research" }));
    expect(ticked()).toEqual(["Outreach"]);
    expect(textbox().value).toBe("Research");
    expect(answer()).toEqual({ selected: ["Outreach"], custom: "Research" });

    await user.type(textbox(), " lead");
    expect(textbox().value).toBe("Research lead");
    expect(ticked()).toEqual(["Outreach"]);
  });

  it("orders the ticks as the options were offered", async () => {
    const user = userEvent.setup();
    render(<FieldHarness initial={empty()} />);

    await user.click(screen.getByRole("checkbox", { name: "Reporting" }));
    await user.click(screen.getByRole("checkbox", { name: "Research" }));

    expect(answer()).toEqual({
      selected: ["Research", "Reporting"],
      custom: "",
    });
  });

  it("keeps a custom answer that equals an option when paging away and back", async () => {
    const user = userEvent.setup();
    render(<SectionHarness />);

    await user.click(screen.getByRole("checkbox", { name: "Outreach" }));
    await user.click(screen.getByRole("button", { name: "Type something…" }));
    await user.type(textbox(), "Research");

    await user.click(screen.getByRole("button", { name: "Go to question 2" }));
    expect(screen.getByText("Which region?")).toBeDefined();
    await user.click(screen.getByRole("button", { name: "Go to question 1" }));

    expect(textbox().value).toBe("Research");
    expect(ticked()).toEqual(["Outreach"]);
    expect(answer()).toEqual({ selected: ["Outreach"], custom: "Research" });
  });
});
