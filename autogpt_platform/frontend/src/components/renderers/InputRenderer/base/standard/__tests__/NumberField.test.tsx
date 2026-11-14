import { fireEvent, screen } from "@testing-library/react";
import { FieldProps } from "@rjsf/utils";
import { describe, expect, it, vi } from "vitest";

import { render } from "@/tests/integrations/test-utils";

import NumberField from "../NumberField";

function StringFieldStub({
  formData,
  onChange,
}: {
  formData?: unknown;
  onChange: (
    value: unknown,
    path: string[],
    errorSchema?: unknown,
    id?: string,
  ) => void;
}) {
  return (
    <input
      aria-label="value"
      value={String(formData ?? "")}
      onChange={(event) =>
        onChange(event.target.value, [], undefined, "root_value")
      }
    />
  );
}

function makeProps(onChange = vi.fn()): FieldProps {
  return {
    id: "root_value",
    name: "value",
    schema: { type: "integer" },
    registry: { fields: { StringField: StringFieldStub } },
    formData: undefined,
    value: undefined,
    onChange,
    onBlur: vi.fn(),
    onFocus: vi.fn(),
    options: {},
    uiSchema: {},
  } as unknown as FieldProps;
}

describe("NumberField", () => {
  it("keeps an integer above 2^53 as the typed string", () => {
    const onChange = vi.fn();
    render(<NumberField {...makeProps(onChange)} />);

    fireEvent.change(screen.getByLabelText("value"), {
      target: { value: "9007199254740993" },
    });

    expect(onChange).toHaveBeenLastCalledWith(
      "9007199254740993",
      [],
      undefined,
      "root_value",
    );
  });

  it("still converts numbers that fit in a float", () => {
    const onChange = vi.fn();
    render(<NumberField {...makeProps(onChange)} />);

    fireEvent.change(screen.getByLabelText("value"), {
      target: { value: "123" },
    });

    expect(onChange).toHaveBeenLastCalledWith(123, [], undefined, "root_value");
  });
});
