import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import {
  MultiSelector,
  MultiSelectorInput,
  MultiSelectorTrigger,
} from "./multiselect";

function renderMultiSelector(onValuesChange = vi.fn()) {
  render(
    <MultiSelector values={["One"]} onValuesChange={onValuesChange}>
      <MultiSelectorTrigger>
        <MultiSelectorInput aria-label="Options" />
      </MultiSelectorTrigger>
    </MultiSelector>,
  );

  return { input: screen.getByLabelText("Options"), onValuesChange };
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

      expect(onValuesChange).toHaveBeenCalledWith([]);
    },
  );
});
