import type { WidgetProps } from "@rjsf/utils";
import type { ReactNode } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { SelectWidget } from "../SelectWidget";

interface SelectMockProps {
  onValueChange?: (value: string) => void;
  options?: unknown;
  value?: string;
}

interface MultiSelectorMockProps {
  children: ReactNode;
  onValuesChange: (values: string[]) => void;
  values: string[];
}

const selectSpy = vi.fn();
const multiSelectorSpy = vi.fn();

vi.mock("@/components/atoms/Select/Select", () => ({
  Select: (props: SelectMockProps) => {
    selectSpy(props);
    return <div data-testid="select-widget-select" />;
  },
}));

vi.mock("@/components/__legacy__/ui/multiselect", () => ({
  MultiSelector: (props: MultiSelectorMockProps) => {
    multiSelectorSpy(props);
    return <div data-testid="multi-selector">{props.children}</div>;
  },
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
  multiSelectorSpy.mockClear();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
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

  it("preserves falsy non-empty values like false", () => {
    render(
      <SelectWidget
        {...createProps({
          schema: { type: "boolean" },
          value: false,
          options: {
            enumOptions: [
              { value: false, label: "Disabled" },
              { value: true, label: "Enabled" },
            ],
          },
        })}
      />,
    );

    expect(selectSpy).toHaveBeenCalledOnce();
    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      value: "0",
      options: [
        { value: "0", label: "Disabled" },
        { value: "1", label: "Enabled" },
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

  it("warns in development when empty-string enum options are dropped", () => {
    vi.stubEnv("NODE_ENV", "development");
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    render(
      <SelectWidget
        {...createProps({
          options: {
            enumOptions: [
              { value: "", label: "Empty" },
              { value: "red", label: "Red" },
            ],
          },
        })}
      />,
    );

    expect(warnSpy).toHaveBeenCalledWith(
      "[SelectWidget] Dropped enum option(s) with empty-string value. Radix Select.Item disallows empty values.",
      {
        schema: { type: "string" },
        dropped: 1,
      },
    );
  });

  it("maps selected indexes back to enum option values for single-select changes", () => {
    const onChange = vi.fn();

    render(
      <SelectWidget
        {...createProps({
          onChange,
          options: {
            enumOptions: [
              { value: "red", label: "Red" },
              { value: "green", label: "Green" },
            ],
          },
        })}
      />,
    );

    const selectProps = selectSpy.mock.calls[0][0] as SelectMockProps;
    selectProps.onValueChange?.("1");

    expect(onChange).toHaveBeenCalledWith("green");
  });

  it("maps multi-select labels back to enum option values", () => {
    const onChange = vi.fn();

    render(
      <SelectWidget
        {...createProps({
          schema: {
            type: "array",
            items: {
              type: "string",
              enum: ["red", "green"],
            },
          },
          value: ["red"],
          onChange,
          options: {
            enumOptions: [
              { value: "red", label: "Red" },
              { value: "green", label: "Green" },
            ],
          },
        })}
      />,
    );

    expect(multiSelectorSpy).toHaveBeenCalledOnce();
    const multiSelectorProps = multiSelectorSpy.mock
      .calls[0][0] as MultiSelectorMockProps;
    expect(multiSelectorProps.values).toEqual(["Red"]);

    multiSelectorProps.onValuesChange(["Red", "Green", "Unknown"]);

    expect(onChange).toHaveBeenCalledWith(["red", "green"]);
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
