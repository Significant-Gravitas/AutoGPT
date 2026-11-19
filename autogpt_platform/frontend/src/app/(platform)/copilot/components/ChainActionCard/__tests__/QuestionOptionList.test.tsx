import { fireEvent, render, screen } from "@testing-library/react";
import type { ComponentProps } from "react";
import { describe, expect, it, vi } from "vitest";
import { QuestionOptionList } from "../QuestionOptionList";

const OPTIONS = ["Europe", "Americas", "Asia"];

function renderList(
  overrides: Partial<ComponentProps<typeof QuestionOptionList>> = {},
) {
  const props = {
    options: OPTIONS,
    value: "",
    labelId: "label-1",
    focusActiveOption: false,
    onChange: vi.fn(),
    onEdit: vi.fn(),
    onSubmit: vi.fn(),
    ...overrides,
  };
  render(
    <>
      <span id="label-1">Which region?</span>
      <QuestionOptionList {...props} />
    </>,
  );
  return props;
}

describe("QuestionOptionList defaults", () => {
  it("renders a pencil per option that copies the option into the editor", () => {
    const { onEdit } = renderList();

    for (const option of OPTIONS) {
      expect(
        screen.getByRole("button", { name: `Edit ${option}` }),
      ).toBeDefined();
    }
    fireEvent.click(screen.getByRole("button", { name: "Edit Americas" }));

    expect(onEdit).toHaveBeenCalledWith("Americas");
  });

  it("reserves room for the pencil on every row", () => {
    renderList();

    for (const option of OPTIONS) {
      const radio = screen.getByRole("radio", { name: option });
      expect(radio.className).toContain("pr-12");
      expect(radio.className).not.toContain("pr-4");
    }
  });

  it("marks the group as required", () => {
    renderList();

    expect(
      screen
        .getByRole("radiogroup", { name: "Which region?" })
        .getAttribute("aria-required"),
    ).toBe("true");
  });
});

describe("QuestionOptionList editable={false}", () => {
  it("hides every pencil", () => {
    renderList({ editable: false });

    expect(screen.queryByRole("button", { name: /^Edit / })).toBeNull();
  });

  it("drops the padding reserved for the pencil", () => {
    renderList({ editable: false, value: "Europe" });

    for (const option of OPTIONS) {
      const radio = screen.getByRole("radio", { name: option });
      expect(radio.className).toContain("pr-4");
      expect(radio.className).not.toContain("pr-12");
    }
  });
});

describe("QuestionOptionList without onEdit", () => {
  it("renders no pencil even while editable", () => {
    renderList({ onEdit: undefined });

    expect(screen.queryByRole("button", { name: /^Edit / })).toBeNull();
    expect(
      screen.getByRole("radio", { name: "Europe" }).className,
    ).not.toContain("pr-12");
  });
});

describe("QuestionOptionList required={false}", () => {
  it("sets aria-required to false", () => {
    renderList({ required: false });

    expect(
      screen
        .getByRole("radiogroup", { name: "Which region?" })
        .getAttribute("aria-required"),
    ).toBe("false");
  });
});

describe("QuestionOptionList selectedContent", () => {
  it("renders the content right after the selected row, before the next one", () => {
    renderList({
      value: "Americas",
      selectedContent: <p>Panel for the choice</p>,
    });

    const group = screen.getByRole("radiogroup", { name: "Which region?" });
    const panel = screen.getByText("Panel for the choice");
    const americas = screen.getByRole("radio", { name: "Americas" });
    const asia = screen.getByRole("radio", { name: "Asia" });

    expect(group.contains(panel)).toBe(true);
    expect(
      americas.compareDocumentPosition(panel) &
        Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(
      panel.compareDocumentPosition(asia) & Node.DOCUMENT_POSITION_FOLLOWING,
    ).toBeTruthy();
    expect(screen.getAllByText("Panel for the choice")).toHaveLength(1);
  });

  it("renders nothing when no option is selected", () => {
    renderList({ selectedContent: <p>Panel for the choice</p> });

    expect(screen.queryByText("Panel for the choice")).toBeNull();
  });

  it("renders nothing when the value matches no option", () => {
    renderList({
      value: "Antarctica",
      selectedContent: <p>Panel for the choice</p>,
    });

    expect(screen.queryByText("Panel for the choice")).toBeNull();
  });
});

describe("QuestionOptionList keyboard", () => {
  it.each([
    ["ArrowDown", "Americas"],
    ["ArrowRight", "Americas"],
    ["ArrowUp", "Asia"],
    ["ArrowLeft", "Asia"],
  ])("%s moves and selects, wrapping around", (key, expected) => {
    const { onChange } = renderList({ value: "Europe", editable: false });

    fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), { key });

    expect(onChange).toHaveBeenCalledWith(expected);
  });

  it("Enter selects an unselected option without submitting", () => {
    const { onChange, onSubmit } = renderList({ editable: false });

    fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
      key: "Enter",
    });

    expect(onChange).toHaveBeenCalledWith("Europe");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("Enter on the selected option submits", () => {
    const { onChange, onSubmit } = renderList({
      value: "Europe",
      editable: false,
    });

    fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
      key: "Enter",
    });

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onChange).not.toHaveBeenCalled();
  });

  it("Space selects without submitting", () => {
    const { onChange, onSubmit } = renderList({
      value: "Europe",
      editable: false,
    });

    fireEvent.keyDown(screen.getByRole("radio", { name: "Asia" }), {
      key: " ",
    });

    expect(onChange).toHaveBeenCalledWith("Asia");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("ignores Enter while an IME is composing", () => {
    const { onChange, onSubmit } = renderList({ value: "Europe" });

    fireEvent.keyDown(screen.getByRole("radio", { name: "Europe" }), {
      key: "Enter",
      isComposing: true,
    });

    expect(onChange).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("keeps a single tab stop on the selected option", () => {
    renderList({ value: "Americas", editable: false });

    expect(
      screen.getByRole("radio", { name: "Americas" }).getAttribute("tabindex"),
    ).toBe("0");
    expect(
      screen.getByRole("radio", { name: "Europe" }).getAttribute("tabindex"),
    ).toBe("-1");
  });
});
