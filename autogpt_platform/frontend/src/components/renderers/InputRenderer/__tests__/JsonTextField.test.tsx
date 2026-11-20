import type { IChangeEvent } from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import { fireEvent, screen, waitFor } from "@testing-library/react";
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
});
