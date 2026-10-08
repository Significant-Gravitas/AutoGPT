import type { WidgetProps } from "@rjsf/utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { SelectWidget } from "../SelectWidget";

interface SelectMockProps {
  label?: string;
  onValueChange?: (value: string) => void;
  options?: unknown;
  placeholder?: string;
  value?: string;
}

interface MultiSelectMockProps {
  options: { value: string; label: string }[];
  onValueChange: (values: string[]) => void;
  value: string[];
}

const selectSpy = vi.fn();
const multiSelectSpy = vi.fn();

function SelectMock(props: SelectMockProps) {
  selectSpy(props);
  return <div data-testid="select-widget-select" />;
}

function MultiSelectMock(props: MultiSelectMockProps) {
  multiSelectSpy(props);
  return <div data-testid="multi-select" />;
}

vi.mock("@/components/atoms/Select/Select", () => ({
  Select: SelectMock,
}));

vi.mock("@/components/molecules/MultiSelect/MultiSelect", () => ({
  MultiSelect: MultiSelectMock,
}));

afterEach(() => {
  selectSpy.mockClear();
  multiSelectSpy.mockClear();
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

  it("forwards an accessible label and placeholder", () => {
    render(
      <SelectWidget
        {...createProps({
          label: "Transport",
          placeholder: "Select a transport",
        })}
      />,
    );

    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      label: "Transport",
      placeholder: "Select a transport",
    });
  });

  it("uses top-level metadata for a nullable enum field", () => {
    const enumSchema: WidgetProps["schema"] = {
      type: "string",
      enum: ["platform", "codex_app_server"],
      enumNames: ["AutoGPT Platform", "ChatGPT"],
      title: "AutoPilotTransport",
    };
    const transportSchema: WidgetProps["schema"] = {
      anyOf: [enumSchema, { type: "null" }],
      placeholder: "Select a transport",
      title: "Transport",
    };

    render(
      <SelectWidget
        {...createProps({
          name: "transport",
          label: "AutoPilotTransport",
          schema: enumSchema,
          registry: {
            ...createProps().registry,
            rootSchema: {
              type: "object",
              properties: { transport: transportSchema },
            },
          },
          options: {
            enumOptions: [
              { value: "platform", label: "platform" },
              { value: "codex_app_server", label: "codex_app_server" },
            ],
          },
        })}
      />,
    );

    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      label: "Transport",
      placeholder: "Select a transport",
      options: [
        { value: "0", label: "AutoGPT Platform" },
        { value: "1", label: "ChatGPT" },
      ],
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
      "[SelectWidget] Dropped enum option(s) with empty-string value. An empty value reads as no selection in Select.",
      {
        schema: { type: "string" },
        dropped: 1,
      },
    );
    expect(selectSpy.mock.calls[0][0]).toMatchObject({
      options: [{ value: "0", label: "Red" }],
    });
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

  it("maps selected indexes back to enum option values for multi-select changes", () => {
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

    expect(multiSelectSpy).toHaveBeenCalledOnce();
    const multiSelectProps = multiSelectSpy.mock
      .calls[0][0] as MultiSelectMockProps;
    expect(multiSelectProps.value).toEqual(["0"]);
    expect(multiSelectProps.options).toEqual([
      { value: "0", label: "Red" },
      { value: "1", label: "Green" },
    ]);

    multiSelectProps.onValueChange(["0", "1", "9"]);

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

    expect(screen.getByTestId("multi-select")).toBeDefined();
    expect(multiSelectSpy.mock.calls[0][0]).toMatchObject({
      options: [{ value: "0", label: "Red" }],
    });
  });
});
