import { describe, expect, test, vi } from "vitest";
import { fireEvent } from "@testing-library/react";
import { render, screen } from "@/tests/integrations/test-utils";
import TextWidget from "./TextWidget";

function makeProps(overrides: Record<string, unknown> = {}) {
  return {
    id: "test-integer-input",
    name: "test-integer-input",
    schema: { type: "integer", title: "Test Integer" },
    onChange: vi.fn(),
    onBlur: vi.fn(),
    onFocus: vi.fn(),
    value: undefined,
    required: false,
    disabled: false,
    readonly: false,
    placeholder: "",
    options: {},
    autofocus: false,
    rawErrors: [],
    label: "",
    hideLabel: false,
    multiple: false,
    formContext: { size: "small", uiType: "default" },
    registry: {
      formContext: { size: "small", uiType: "default" },
      fields: {},
      widgets: {},
      rootSchema: {},
      schemaUtils: {},
      templates: {},
      translateString: (key: string) => key,
    },
    ...overrides,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  } as any;
}

describe("TextWidget — INTEGER input", () => {
  test("renders the integer input with htmlType=number", () => {
    render(<TextWidget {...makeProps()} />);
    const input = screen.getByPlaceholderText(/Enter integer value/i);
    expect(input.getAttribute("type")).toBe("number");
  });

  test("truncates decimal input via Math.trunc when calling onChange", () => {
    const onChange = vi.fn();
    render(<TextWidget {...makeProps({ onChange })} />);
    const input = screen.getByPlaceholderText(/Enter integer value/i);
    fireEvent.change(input, { target: { value: "12.7" } });
    expect(onChange).toHaveBeenCalledWith(12);
  });

  test("returns undefined when the value is cleared", () => {
    const onChange = vi.fn();
    render(<TextWidget {...makeProps({ value: 42, onChange })} />);
    const input = screen.getByPlaceholderText(/Enter integer value/i);
    fireEvent.change(input, { target: { value: "" } });
    expect(onChange).toHaveBeenCalledWith(undefined);
  });
});
