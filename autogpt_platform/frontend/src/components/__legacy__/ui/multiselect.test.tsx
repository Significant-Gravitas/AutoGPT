import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  MultiSelector,
  MultiSelectorContent,
  MultiSelectorInput,
  MultiSelectorTrigger,
} from "./multiselect";

function renderMultiSelector({
  dir,
  loop,
  onValuesChange = vi.fn(),
}: {
  dir?: "ltr" | "rtl";
  loop?: boolean;
  onValuesChange?: (value: string[]) => void;
} = {}) {
  render(
    <MultiSelector
      values={["One", "Two"]}
      onValuesChange={onValuesChange}
      dir={dir}
      loop={loop}
    >
      <MultiSelectorTrigger>
        <MultiSelectorInput aria-label="Options" />
      </MultiSelectorTrigger>
      <MultiSelectorContent>
        <div>Choices</div>
      </MultiSelectorContent>
    </MultiSelector>,
  );

  return { input: screen.getByLabelText("Options"), onValuesChange };
}

function isActive(name: string) {
  return screen.getByText(name).parentElement?.classList.contains("ring-2");
}

describe("MultiSelector keyboard handling", () => {
  it.each(["Backspace", "Delete"])(
    "does not remove a value with %s while an IME is composing",
    (key) => {
      const { input, onValuesChange } = renderMultiSelector();

      fireEvent.keyDown(input, { key, isComposing: true });

      expect(onValuesChange).not.toHaveBeenCalled();
    },
  );

  it.each(["Backspace", "Delete"])(
    "removes the final value with a plain %s keydown",
    (key) => {
      const { input, onValuesChange } = renderMultiSelector();

      fireEvent.keyDown(input, { key });

      expect(onValuesChange).toHaveBeenCalledWith(["One"]);
    },
  );

  it("keeps values while deleting text from the filter", () => {
    const { input, onValuesChange } = renderMultiSelector();

    fireEvent.change(input, { target: { value: "filter" } });
    fireEvent.keyDown(input, { key: "Backspace" });

    expect(onValuesChange).not.toHaveBeenCalled();
  });

  it("opens with Enter and closes with Escape", () => {
    const { input } = renderMultiSelector();

    fireEvent.keyDown(input, { key: "Enter" });
    expect(screen.getByText("Choices")).toBeDefined();

    fireEvent.keyDown(input, { key: "Escape" });
    expect(screen.queryByText("Choices")).toBeNull();
  });

  it("ignores Enter and Escape while an IME is composing", () => {
    const { input } = renderMultiSelector();

    fireEvent.keyDown(input, { key: "Enter", isComposing: true });
    expect(screen.queryByText("Choices")).toBeNull();

    fireEvent.focus(input);
    expect(screen.getByText("Choices")).toBeDefined();

    fireEvent.keyDown(input, { key: "Escape", isComposing: true });
    expect(screen.getByText("Choices")).toBeDefined();
  });

  it.each([
    { dir: "ltr" as const, key: "ArrowLeft" },
    { dir: "rtl" as const, key: "ArrowRight" },
  ])("selects the final value with $dir $key", ({ dir, key }) => {
    const { input } = renderMultiSelector({ dir });

    fireEvent.keyDown(input, { key });

    expect(isActive("Two")).toBe(true);
    expect(isActive("One")).toBe(false);
  });

  it.each([
    { dir: "ltr" as const, back: "ArrowLeft", forward: "ArrowRight" },
    { dir: "rtl" as const, back: "ArrowRight", forward: "ArrowLeft" },
  ])(
    "wraps from the final value to the first with $dir $forward when looping",
    ({ dir, back, forward }) => {
      const { input } = renderMultiSelector({ dir, loop: true });

      fireEvent.keyDown(input, { key: back });
      expect(isActive("Two")).toBe(true);

      fireEvent.keyDown(input, { key: forward });
      expect(isActive("One")).toBe(true);
      expect(isActive("Two")).toBe(false);
    },
  );

  it("clears the selection past the final value without looping", () => {
    const { input } = renderMultiSelector();

    fireEvent.keyDown(input, { key: "ArrowLeft" });
    fireEvent.keyDown(input, { key: "ArrowRight" });

    expect(isActive("One")).toBe(false);
    expect(isActive("Two")).toBe(false);
  });

  it("moves forward from an active value in RTL mode", () => {
    const { input } = renderMultiSelector({ dir: "rtl" });

    fireEvent.keyDown(input, { key: "ArrowRight" });
    fireEvent.keyDown(input, { key: "ArrowRight" });
    expect(isActive("One")).toBe(true);

    fireEvent.keyDown(input, { key: "ArrowLeft" });
    expect(isActive("Two")).toBe(true);
    expect(isActive("One")).toBe(false);
  });

  it("does not move forward from no active value without looping", () => {
    const { input } = renderMultiSelector({ dir: "rtl" });

    fireEvent.keyDown(input, { key: "ArrowLeft" });

    expect(isActive("One")).toBe(false);
    expect(isActive("Two")).toBe(false);
  });
});
