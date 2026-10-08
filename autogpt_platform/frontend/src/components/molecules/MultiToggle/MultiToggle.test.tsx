import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MultiToggle } from "./MultiToggle";

const ITEMS = [
  { value: "cat", label: "猫" },
  { value: "dog", label: "犬" },
];

function renderToggle(selectedValues: string[] = []) {
  const onChange = vi.fn();
  render(
    <MultiToggle
      items={ITEMS}
      selectedValues={selectedValues}
      onChange={onChange}
      aria-label="Animals"
    />,
  );
  return { onChange, cat: screen.getByRole("button", { name: "猫" }) };
}

describe("MultiToggle", () => {
  it("exposes the selection as a labelled group of pressed buttons", () => {
    const { cat } = renderToggle(["dog"]);

    expect(screen.getByRole("group", { name: "Animals" })).toBeDefined();
    expect(cat.getAttribute("aria-pressed")).toBe("false");
    expect(
      screen.getByRole("button", { name: "犬" }).getAttribute("aria-pressed"),
    ).toBe("true");
  });

  it("selects an item on click", () => {
    const { onChange, cat } = renderToggle();

    fireEvent.click(cat);

    expect(onChange).toHaveBeenCalledWith(["cat"]);
  });

  it("deselects an already selected item", () => {
    const { onChange, cat } = renderToggle(["cat", "dog"]);

    fireEvent.click(cat);

    expect(onChange).toHaveBeenCalledWith(["dog"]);
  });

  it.each([
    ["a composing Space", { key: " ", isComposing: true }],
    ["a composing Enter", { key: "Enter", isComposing: true }],
    ["Safari's post-composition Enter", { key: "Enter", keyCode: 229 }],
  ])("leaves the selection alone on %s", (_label, init) => {
    const { onChange, cat } = renderToggle();

    fireEvent.keyDown(cat, init);

    expect(onChange).not.toHaveBeenCalled();
  });

  it("ignores unrelated keys", () => {
    const { onChange, cat } = renderToggle();

    fireEvent.keyDown(cat, { key: "a" });

    expect(onChange).not.toHaveBeenCalled();
  });
});
