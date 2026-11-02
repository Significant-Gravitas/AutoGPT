import { fireEvent, screen } from "@testing-library/react";
import { WidgetProps } from "@rjsf/utils";
import { describe, expect, it, vi } from "vitest";

import { BlockUIType } from "@/app/(platform)/build/components/types";
import { render } from "@/tests/integrations/test-utils";

import TextWidget from "../TextWidget";

function makeProps(overrides: Partial<WidgetProps> = {}): WidgetProps {
  const defaults = {
    id: "test-input",
    name: "test-input",
    label: "Test",
    value: undefined,
    onChange: vi.fn(),
    onBlur: vi.fn(),
    onFocus: vi.fn(),
    schema: { type: "string" },
    options: {},
    required: false,
    disabled: false,
    readonly: false,
    registry: {
      formContext: {
        size: "small",
        uiType: BlockUIType.STANDARD,
      },
    },
  };

  return {
    ...defaults,
    ...overrides,
    registry: {
      ...defaults.registry,
      ...(overrides.registry ?? {}),
    },
  } as unknown as WidgetProps;
}

describe("TextWidget", () => {
  describe("integer input", () => {
    it("parses plain integer strings", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "integer" },
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "123" },
      });

      expect(onChange).toHaveBeenLastCalledWith(123);
    });

    it("returns undefined for empty string", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "integer" },
        value: 5,
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "" },
      });

      expect(onChange).toHaveBeenLastCalledWith(undefined);
    });

    it("returns undefined for non-numeric value rather than NaN", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "integer" },
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "1,234" },
      });

      expect(onChange).toHaveBeenLastCalledWith(undefined);
      expect(onChange).not.toHaveBeenCalledWith(NaN);
    });

    it("truncates fractional values to integer", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "integer" },
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "1.7" },
      });

      expect(onChange).toHaveBeenLastCalledWith(1);
    });

    it("correctly handles scientific notation", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "integer" },
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "1e5" },
      });

      expect(onChange).toHaveBeenLastCalledWith(100000);
    });

    it("rejects non-finite values like 1e309", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "integer" },
        value: 1,
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "1e309" },
      });

      expect(onChange).toHaveBeenLastCalledWith(undefined);
    });

    it("uses HTML number input type", () => {
      const props = makeProps({ schema: { type: "integer" } });
      render(<TextWidget {...props} />);
      expect(screen.getByRole("spinbutton").getAttribute("type")).toBe(
        "number",
      );
    });

    it("displays empty string when value is NaN", () => {
      const props = makeProps({
        schema: { type: "integer" },
        value: NaN,
      });

      render(<TextWidget {...props} />);

      const input = screen.getByRole("spinbutton") as HTMLInputElement;
      expect(input.value).toBe("");
    });

    it("displays empty string when value is Infinity", () => {
      const props = makeProps({
        schema: { type: "integer" },
        value: Infinity,
      });

      render(<TextWidget {...props} />);

      const input = screen.getByRole("spinbutton") as HTMLInputElement;
      expect(input.value).toBe("");
    });
  });

  describe("number input", () => {
    it("parses decimal values", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "number" },
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "1.5" },
      });

      expect(onChange).toHaveBeenLastCalledWith(1.5);
    });

    it("returns undefined for non-numeric value rather than NaN", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "number" },
        value: 1,
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("spinbutton"), {
        target: { value: "abc" },
      });

      expect(onChange).toHaveBeenLastCalledWith(undefined);
      expect(onChange).not.toHaveBeenCalledWith(NaN);
    });
  });

  describe("string input", () => {
    it("passes value through unchanged", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "string" },
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("textbox"), {
        target: { value: "hello" },
      });

      expect(onChange).toHaveBeenLastCalledWith("hello");
    });

    it("returns undefined for empty string", () => {
      const onChange = vi.fn();
      const props = makeProps({
        schema: { type: "string" },
        value: "abc",
        onChange,
      });

      render(<TextWidget {...props} />);

      fireEvent.change(screen.getByRole("textbox"), {
        target: { value: "" },
      });

      expect(onChange).toHaveBeenLastCalledWith(undefined);
    });
  });
});
