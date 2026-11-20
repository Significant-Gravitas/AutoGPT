import type { IChangeEvent } from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { BlockUIType } from "@/app/(platform)/build/components/types";
import { render } from "@/tests/integrations/test-utils";
import { expectTypingKept } from "@/tests/integrations/typing";

import { FormRenderer } from "../FormRenderer";

interface RowsData {
  rows: Record<string, unknown>[];
}

// Free-form dicts inside an array are edited as raw JSON.
const schema: RJSFSchema = {
  type: "object",
  properties: {
    rows: {
      title: "Rows",
      type: "array",
      items: { type: "object", additionalProperties: true },
    },
  },
};

function RowsForm({ rows }: { rows: Record<string, unknown>[] }) {
  const [data, setData] = useState<RowsData>({ rows });

  function handleChange(event: IChangeEvent<RowsData>) {
    if (event.formData) setData(event.formData);
  }

  return (
    <>
      <FormRenderer
        jsonSchema={schema}
        handleChange={handleChange}
        uiSchema={{}}
        initialValues={data}
        formContext={{
          nodeId: "json-node",
          uiType: BlockUIType.STANDARD,
          showHandles: false,
          size: "small",
        }}
      />
      <output aria-label="Stored rows">{JSON.stringify(data.rows)}</output>
      <button type="button" onClick={() => setData({ rows: [{ b: 2 }] })}>
        Load other rows
      </button>
    </>
  );
}

const LONG_NUMBER_HINT = /put them in quotes to keep every digit/i;

function getJsonField() {
  return screen.getByPlaceholderText('{"key": "value"}') as HTMLTextAreaElement;
}

describe("JsonTextField", () => {
  it("keeps the JSON as typed instead of reformatting it", async () => {
    render(<RowsForm rows={[{}]} />);

    await expectTypingKept(getJsonField(), '"a":1', 1);

    expect(screen.getByLabelText("Stored rows").textContent).toBe('[{"a":1}]');
  });

  it("shows data that changes from outside the field", async () => {
    render(<RowsForm rows={[{ a: 1 }]} />);
    expect(getJsonField().value).toBe('{\n  "a": 1\n}');

    fireEvent.click(screen.getByRole("button", { name: "Load other rows" }));

    await waitFor(() => expect(getJsonField().value).toBe('{\n  "b": 2\n}'));
  });

  it("keeps typing next to a number stored as a rounded big integer", async () => {
    render(<RowsForm rows={[{ id: 12345678901234567000 }]} />);
    const field = getJsonField();

    await expectTypingKept(field, ', "a": 1', field.value.indexOf("\n}"));
  });

  it("keeps a negative decimal while it is typed", async () => {
    render(<RowsForm rows={[{}]} />);

    await expectTypingKept(getJsonField(), '"a":-0.5', 1);
  });

  it("tells the user to quote a whole number too long to keep exactly", async () => {
    render(<RowsForm rows={[{}]} />);
    const field = getJsonField();
    expect(screen.queryByText(LONG_NUMBER_HINT)).toBeNull();

    await expectTypingKept(field, '"channel_id": 12345678901234', 1);
    expect(screen.queryByText(LONG_NUMBER_HINT)).toBeNull();

    fireEvent.change(field, {
      target: { value: '{"channel_id": 1234567890123456789}' },
    });

    const hint = await screen.findByText(LONG_NUMBER_HINT);
    expect(field.getAttribute("aria-describedby")?.split(" ")).toContain(
      hint.id,
    );
  });

  it("shows no hint for a long number in quotes or a safe one", () => {
    render(<RowsForm rows={[{}]} />);
    const field = getJsonField();

    fireEvent.change(field, {
      target: { value: '{"channel_id": "1234567890123456789"}' },
    });
    expect(screen.queryByText(LONG_NUMBER_HINT)).toBeNull();

    fireEvent.change(field, {
      target: { value: '{"channel_id": 9007199254740991}' },
    });
    expect(screen.queryByText(LONG_NUMBER_HINT)).toBeNull();
  });

  it("shows the hint in the expanded editor too", async () => {
    render(<RowsForm rows={[{}]} />);

    fireEvent.click(screen.getByRole("button", { name: "Expand JSON input" }));
    const dialog = await screen.findByRole("dialog");
    const editor = within(dialog).getByRole("textbox");

    fireEvent.change(editor, { target: { value: "[1234567890123456789]" } });
    expect(within(dialog).getByText(LONG_NUMBER_HINT)).toBeDefined();

    fireEvent.change(editor, { target: { value: '["1234567890123456789"]' } });
    expect(within(dialog).queryByText(LONG_NUMBER_HINT)).toBeNull();
  });
});
