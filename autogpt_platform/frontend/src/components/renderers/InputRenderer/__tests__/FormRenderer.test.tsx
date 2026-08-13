import { fireEvent, screen, waitFor } from "@testing-library/react";
import { RJSFSchema, WidgetProps } from "@rjsf/utils";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import { BlockUIType } from "@/app/(platform)/build/components/types";
import { render } from "@/tests/integrations/test-utils";

import { FormRenderer } from "../FormRenderer";
import { SelectWidget } from "../base/standard/widgets/SelectInput/SelectWidget";

vi.mock("@/components/atoms/Select/Select", () => ({
  Select: ({
    id,
    value,
    onValueChange,
    options,
  }: {
    id: string;
    value?: string;
    onValueChange?: (value: string) => void;
    options: { value: string; label: string }[];
  }) => (
    <select
      aria-label={id}
      value={value ?? ""}
      onChange={(event) => onValueChange?.(event.target.value)}
    >
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
}));

let mockMultiSelectValues: string[] = [];
let mockMultiSelectOnChange: (values: string[]) => void = () => undefined;

vi.mock("@/components/__legacy__/ui/multiselect", () => ({
  MultiSelector: ({
    values,
    onValuesChange,
    children,
  }: {
    values: string[];
    onValuesChange: (values: string[]) => void;
    children: React.ReactNode;
  }) => (
    <div>
      {(() => {
        mockMultiSelectValues = values;
        mockMultiSelectOnChange = onValuesChange;
        return children;
      })()}
    </div>
  ),
  MultiSelectorTrigger: ({ children }: { children: React.ReactNode }) => {
    return (
      <div>
        <div data-testid="selected-values">
          {mockMultiSelectValues.join(",")}
        </div>
        {children}
      </div>
    );
  },
  MultiSelectorInput: () => <input aria-label="multi-select-input" />,
  MultiSelectorContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  MultiSelectorList: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  MultiSelectorItem: ({
    value,
    children,
  }: {
    value: string;
    children: React.ReactNode;
  }) => {
    return (
      <button
        type="button"
        onClick={() =>
          mockMultiSelectOnChange([...mockMultiSelectValues, value])
        }
      >
        {children}
      </button>
    );
  },
}));

describe("FormRenderer", () => {
  it("round-trips numeric enum values for array item selects", async () => {
    const handleChange = vi.fn();
    const jsonSchema: RJSFSchema = {
      type: "object",
      properties: {
        reminder_minutes: {
          title: "Reminder Minutes",
          type: "array",
          items: {
            title: "ReminderPreset",
            type: "integer",
            enum: [10, 30, 60, 1440],
          },
        },
      },
    };

    render(
      <FormRenderer
        jsonSchema={jsonSchema}
        handleChange={handleChange}
        uiSchema={{}}
        initialValues={{ reminder_minutes: [] }}
        formContext={{
          nodeId: "node-1",
          uiType: BlockUIType.STANDARD,
          showHandles: false,
          size: "small",
        }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: /add item/i }));

    const select = screen.getByRole("combobox") as HTMLSelectElement;
    expect(Array.from(select.options).map((option) => option.text)).toEqual([
      "10",
      "30",
      "60",
      "1440",
    ]);

    fireEvent.change(select, { target: { value: "2" } });

    await waitFor(() => {
      const lastCall = handleChange.mock.calls.at(-1)?.[0];
      expect(lastCall.formData.reminder_minutes).toEqual([60]);
    });
  });

  it("round-trips numeric enum values for multiselects without exposing indexes", () => {
    const onChange = vi.fn();

    render(
      <SelectWidget
        id="reminder-minutes"
        name="reminder-minutes"
        label="Reminder Minutes"
        schema={{
          type: "array",
          items: {
            type: "integer",
            enum: [10, 30, 60, 1440],
          },
        }}
        options={{
          enumOptions: [
            { value: 10, label: "10" },
            { value: 30, label: "30" },
            { value: 60, label: "60" },
            { value: 1440, label: "1440" },
          ],
        }}
        value={[10, 60]}
        onChange={onChange}
        onBlur={vi.fn()}
        onFocus={vi.fn()}
        disabled={false}
        readonly={false}
        formContext={{ size: "small" }}
        registry={{} as WidgetProps["registry"]}
      />,
    );

    expect(screen.getByTestId("selected-values").textContent).toBe("10,60");

    fireEvent.click(screen.getByRole("button", { name: "30" }));

    expect(onChange).toHaveBeenCalledWith([10, 60, 30]);
  });
});
