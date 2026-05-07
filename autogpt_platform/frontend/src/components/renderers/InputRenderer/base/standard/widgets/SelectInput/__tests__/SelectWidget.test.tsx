import { WidgetProps } from "@rjsf/utils";
import { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { SelectWidget } from "../SelectWidget";

const selectSpy = vi.fn();

vi.mock("@/components/atoms/Select/Select", () => ({
  Select: (props: Record<string, unknown>) => {
    selectSpy(props);
    return <div data-testid="select-widget-select" />;
  },
}));

vi.mock("@/components/__legacy__/ui/multiselect", () => ({
  MultiSelector: ({ children }: { children: ReactNode }) => (
    <div data-testid="multi-selector">{children}</div>
  ),
  MultiSelectorContent: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  MultiSelectorInput: ({ placeholder }: { placeholder?: string }) => (
    <input data-testid="multi-selector-input" placeholder={placeholder} />
  ),
  MultiSelectorItem: ({
    children,
    value,
  }: {
    children: ReactNode;
    value: string;
  }) => (
    <div data-testid="multi-selector-item" data-value={value}>
      {children}
    </div>
  ),
  MultiSelectorList: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
  MultiSelectorTrigger: ({ children }: { children: ReactNode }) => (
    <div>{children}</div>
  ),
}));

afterEach(() => {
  cleanup();
  selectSpy.mockClear();
});

function createProps(overrides: Partial<WidgetProps> = {}): WidgetProps {
  return {
    id: "color",
    name: "color",
    schema: { type: "string" },
    uiSchema: {},
    value: undefined,
    required: false,
    disabled: false,
    readonly: false,
    hideError: false,
    autofocus: false,
    label: "",
    options: { enumOptions: [] },
    formContext: {},
    onChange: vi.fn(),
    onBlur: vi.fn(),
    onFocus: vi.fn(),
    rawErrors: [],
    registry: {} as WidgetProps["registry"],
    ...overrides,
  };
}

describe("SelectWidget", () => {
  it("passes an empty selected value to Select when the current value is an empty string", () => {
    render(
      <SelectWidget
        {...createProps({
          value: "",
          options: {
            enumOptions: [{ value: "red", label: "Red" }],
          },
        })}
      />,
    );

    expect(selectSpy).toHaveBeenCalledOnce();
    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      value: "",
      options: [{ value: "0", label: "Red" }],
    });
  });

  it("preserves falsy non-empty values like 0", () => {
    render(
      <SelectWidget
        {...createProps({
          value: 0,
          options: {
            enumOptions: [
              { value: 0, label: "Zero" },
              { value: 1, label: "One" },
            ],
          },
        })}
      />,
    );

    expect(selectSpy).toHaveBeenCalledOnce();
    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      value: "0",
      options: [
        { value: "0", label: "Zero" },
        { value: "1", label: "One" },
      ],
    });
  });

  it("falls back to an empty option list when enumOptions are missing", () => {
    render(
      <SelectWidget
        {...createProps({
          options: {} as WidgetProps["options"],
        })}
      />,
    );

    expect(selectSpy).toHaveBeenCalledOnce();
    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      options: [],
    });
  });

  it("filters empty-string enum options for the multi-select path", () => {
    render(
      <SelectWidget
        {...createProps({
          schema: {
            type: "array",
            items: {
              type: "string",
              enum: ["", "red"],
            },
          },
          value: [],
          options: {
            enumOptions: [
              { value: "", label: "Empty" },
              { value: "red", label: "Red" },
            ],
          },
        })}
      />,
    );

    const items = screen.getAllByTestId("multi-selector-item");
    expect(items).toHaveLength(1);
    expect(items[0].getAttribute("data-value")).toBe("Red");
    expect(items[0].textContent).toContain("Red");
  });
});
