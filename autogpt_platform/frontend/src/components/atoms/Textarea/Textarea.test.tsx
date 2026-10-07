import { fireEvent, render, screen } from "@testing-library/react";
import { createRef, useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Textarea } from "./Textarea";

describe("Textarea", () => {
  it("labels the field without using the label as placeholder", () => {
    render(<Textarea label="Comment" />);

    const field = screen.getByRole("textbox", { name: "Comment" });
    expect(field.tagName).toBe("TEXTAREA");
    expect(field.getAttribute("placeholder")).toBeNull();
    expect(field.getAttribute("rows")).toBe("3");
  });

  it("keeps a hidden label as the accessible name", () => {
    render(<Textarea label="Comment" hideLabel />);

    expect(screen.getByRole("textbox", { name: "Comment" })).toBeDefined();
    expect(screen.getByText("Comment").closest("label")?.className).toContain(
      "sr-only",
    );
  });

  it("links hint and error and marks the field invalid", () => {
    render(
      <Textarea
        id="notes"
        label="Notes"
        hint="Optional"
        error="Too short"
        aria-describedby="external"
      />,
    );

    const field = screen.getByRole("textbox", { name: "Notes" });
    expect(field.getAttribute("aria-invalid")).toBe("true");
    expect(field.getAttribute("aria-describedby")).toBe(
      "external notes-hint notes-error",
    );
  });

  it("counts characters when uncontrolled", () => {
    render(<Textarea label="Bio" maxLength={10} defaultValue="abc" />);

    expect(screen.getByText("3/10")).toBeDefined();

    fireEvent.change(screen.getByRole("textbox", { name: "Bio" }), {
      target: { value: "abcdefghij" },
    });

    expect(screen.getByText("10/10")).toBeDefined();
  });

  it("counts characters when controlled and still calls onChange", () => {
    const onChange = vi.fn();
    function Controlled() {
      const [value, setValue] = useState("hi");
      return (
        <Textarea
          label="Bio"
          maxLength={20}
          value={value}
          onChange={(event) => {
            setValue(event.target.value);
            onChange(event.target.value);
          }}
        />
      );
    }
    render(<Controlled />);

    expect(screen.getByText("2/20")).toBeDefined();

    fireEvent.change(screen.getByRole("textbox", { name: "Bio" }), {
      target: { value: "hello" },
    });

    expect(onChange).toHaveBeenCalledWith("hello");
    expect(screen.getByText("5/20")).toBeDefined();
  });

  it("hides the counter when showCount is false", () => {
    render(<Textarea label="Bio" maxLength={10} showCount={false} />);

    expect(screen.queryByText("0/10")).toBeNull();
  });

  it("drops keydowns an IME is composing", () => {
    const onKeyDown = vi.fn();
    render(<Textarea label="Message" onKeyDown={onKeyDown} />);
    const field = screen.getByRole("textbox", { name: "Message" });

    fireEvent.keyDown(field, { key: "Enter", isComposing: true });
    expect(onKeyDown).not.toHaveBeenCalled();

    fireEvent.keyDown(field, { key: "Enter" });
    expect(onKeyDown).toHaveBeenCalledTimes(1);
  });

  it("passes rows, name and size through and forwards its ref", () => {
    const ref = createRef<HTMLTextAreaElement>();
    render(
      <Textarea ref={ref} label="Notes" rows={6} name="notes" size="sm" />,
    );

    expect(ref.current).toBe(screen.getByRole("textbox", { name: "Notes" }));
    expect(ref.current?.getAttribute("rows")).toBe("6");
    expect(ref.current?.getAttribute("name")).toBe("notes");
    expect(ref.current?.className).toContain("px-3");
  });
});
