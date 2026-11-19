import { fireEvent, render, screen } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { expectTypingKept } from "@/tests/integrations/typing";
import { Input } from "./Input";

describe("Input keyboard handling", () => {
  it("forwards a plain Enter keydown to onKeyDown", () => {
    const onKeyDown = vi.fn();
    render(<Input id="name" label="Name" onKeyDown={onKeyDown} />);

    fireEvent.keyDown(screen.getByLabelText("Name"), { key: "Enter" });

    expect(onKeyDown).toHaveBeenCalledTimes(1);
  });

  it("swallows keydowns fired while an IME is composing", () => {
    const onKeyDown = vi.fn();
    render(<Input id="name" label="Name" onKeyDown={onKeyDown} />);

    fireEvent.keyDown(screen.getByLabelText("Name"), {
      key: "Enter",
      isComposing: true,
    });

    expect(onKeyDown).not.toHaveBeenCalled();
  });

  it("guards the textarea variant the same way", () => {
    const onKeyDown = vi.fn();
    render(
      <Input id="bio" label="Bio" type="textarea" onKeyDown={onKeyDown} />,
    );
    const textarea = screen.getByLabelText("Bio");

    fireEvent.keyDown(textarea, { key: "Enter", isComposing: true });
    expect(onKeyDown).not.toHaveBeenCalled();

    fireEvent.keyDown(textarea, { key: "Enter" });
    expect(onKeyDown).toHaveBeenCalledTimes(1);
  });

  it("guards and forwards keydowns for the amount variant", () => {
    const onKeyDown = vi.fn();
    render(
      <Input id="amount" label="Amount" type="amount" onKeyDown={onKeyDown} />,
    );
    const amount = screen.getByLabelText("Amount");

    fireEvent.keyDown(amount, { key: "Enter", isComposing: true });
    expect(onKeyDown).not.toHaveBeenCalled();

    fireEvent.keyDown(amount, { key: "Enter" });
    expect(onKeyDown).toHaveBeenCalledTimes(1);
  });
});

// Every text field built on the atom inherits this: if the atom ever rewrote
// its value (a formatter, a late sync), IME input and the caret would break
// everywhere at once.
describe("Input typing", () => {
  function ControlledInput({ type }: { type: "text" | "textarea" }) {
    const [value, setValue] = useState("Wstern");
    return (
      <Input
        id="field"
        label="Field"
        type={type}
        value={value}
        onChange={(e) => setValue(e.target.value)}
      />
    );
  }

  it.each(["text", "textarea"] as const)(
    "keeps the text and caret as typed in the %s variant",
    async (type) => {
      render(<ControlledInput type={type} />);
      const field = screen.getByLabelText("Field") as HTMLInputElement;

      await expectTypingKept(field, "e", 1);
      await expectTypingKept(field, " 日本語");
    },
  );
});

describe("Input textSize", () => {
  it("keeps the small size's text-body by default", () => {
    render(<Input id="q" label="Search" hideLabel size="small" />);
    const classes = screen.getByRole("textbox").className.split(" ");
    expect(classes).toContain("text-body");
    expect(classes).toContain("placeholder:text-body");
  });

  it("sets the value at text-body-md for body-md, placeholder unchanged", () => {
    render(
      <Input id="q" label="Search" hideLabel size="small" textSize="body-md" />,
    );
    const classes = screen.getByRole("textbox").className.split(" ");
    expect(classes).toContain("text-body-md");
    expect(classes).not.toContain("text-body");
    expect(classes).toContain("placeholder:text-body");
  });

  it("leaves a medium input's size unchanged by default", () => {
    render(<Input id="q" label="Search" hideLabel />);
    const classes = screen.getByRole("textbox").className.split(" ");
    expect(classes).not.toContain("text-body-md");
  });

  it("sets text-body-md on medium text and amount inputs for body-md", () => {
    render(
      <>
        <Input id="q" label="Search" hideLabel textSize="body-md" />
        <Input
          id="amount"
          label="Amount"
          hideLabel
          type="amount"
          textSize="body-md"
        />
      </>,
    );
    for (const box of screen.getAllByRole("textbox")) {
      expect(box.className.split(" ")).toContain("text-body-md");
    }
  });
});
