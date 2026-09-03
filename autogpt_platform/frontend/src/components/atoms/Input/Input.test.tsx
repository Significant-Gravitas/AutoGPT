import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
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
});
