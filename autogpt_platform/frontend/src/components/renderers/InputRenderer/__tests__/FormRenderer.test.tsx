import { fireEvent, screen, waitFor } from "@testing-library/react";
import { RJSFSchema } from "@rjsf/utils";
import { describe, expect, it, vi } from "vitest";

import { BlockUIType } from "@/app/(platform)/build/components/types";
import { render } from "@/tests/integrations/test-utils";

import { FormRenderer } from "../FormRenderer";

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
});
