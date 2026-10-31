import type { IChangeEvent } from "@rjsf/core";
import type { RJSFSchema } from "@rjsf/utils";
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { useState } from "react";
import { describe, expect, it } from "vitest";

import { BlockUIType } from "@/app/(platform)/build/components/types";
import { render } from "@/tests/integrations/test-utils";

import { FormRenderer } from "../FormRenderer";

interface DictionaryData {
  prompt_values: Record<string, string>;
}

const schema: RJSFSchema = {
  type: "object",
  properties: {
    prompt_values: {
      title: "Prompt values",
      type: "object",
      additionalProperties: { type: "string" },
    },
  },
};

function DictionaryForm({ values }: { values: Record<string, string> }) {
  const [data, setData] = useState<DictionaryData>({ prompt_values: values });

  function handleChange(event: IChangeEvent<DictionaryData>) {
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
          nodeId: "dictionary-node",
          uiType: BlockUIType.STANDARD,
          showHandles: false,
          size: "small",
        }}
      />
      <output aria-label="Stored dictionary">{JSON.stringify(data)}</output>
    </>
  );
}

function readDictionary(): Record<string, string> {
  return JSON.parse(
    screen.getByLabelText("Stored dictionary").textContent || "{}",
  ).prompt_values;
}

describe("Dictionary input preservation", () => {
  it("adds a new key containing an existing key without dropping either entry", async () => {
    render(<DictionaryForm values={{ "PR-Title": "First" }} />);

    fireEvent.click(screen.getByRole("button", { name: /add item/i }));
    const newKey = screen.getByDisplayValue("newKey");
    for (const value of ["PR", "PR-Title", "PR-Title-test"]) {
      fireEvent.change(newKey, { target: { value } });
      expect(screen.getByDisplayValue("First")).toBeDefined();
    }
    fireEvent.blur(newKey);

    await waitFor(() => {
      expect(Object.keys(readDictionary())).toEqual([
        "PR-Title",
        "PR-Title-test",
      ]);
      expect(readDictionary()["PR-Title"]).toBe("First");
      expect(screen.getByDisplayValue("PR-Title-test")).toBeDefined();
    });
  });

  it("keeps a populated entry when its key is cleared and blurred", async () => {
    render(<DictionaryForm values={{ title: "Keep this value" }} />);

    const key = screen.getByDisplayValue("title");
    fireEvent.change(key, { target: { value: "" } });
    fireEvent.blur(key);

    await waitFor(() => {
      expect(readDictionary()).toEqual({ title: "Keep this value" });
      expect(screen.getByDisplayValue("Keep this value")).toBeDefined();
      expect(screen.getByDisplayValue("title")).toBeDefined();
    });
  });

  it("renames an entry without losing its value", async () => {
    render(<DictionaryForm values={{ title: "Keep this value" }} />);

    const key = screen.getByDisplayValue("title");
    fireEvent.change(key, { target: { value: "new-title" } });
    fireEvent.blur(key);

    await waitFor(() => {
      expect(readDictionary()).toEqual({ "new-title": "Keep this value" });
    });
  });

  it("removes an entry only through its remove action", async () => {
    render(<DictionaryForm values={{ title: "Remove this value" }} />);

    fireEvent.click(screen.getByRole("button", { name: /^remove$/i }));

    await waitFor(() => {
      expect(readDictionary()).toEqual({});
      expect(screen.queryByDisplayValue("Remove this value")).toBeNull();
    });
  });

  it.each([
    ["PR-Title", "PR-Title-test"],
    ["key[0]", "key[0]-suffix"],
    ["cost$&", "cost$&-suffix"],
    ["cost$$", "cost$$-suffix"],
  ])("preserves both values while editing %s and %s", async (first, second) => {
    render(
      <DictionaryForm values={{ [first]: "First", [second]: "Second" }} />,
    );

    fireEvent.change(screen.getByDisplayValue("Second"), {
      target: { value: "Updated second" },
    });

    await waitFor(() => {
      expect(readDictionary()).toEqual({
        [first]: "First",
        [second]: "Updated second",
      });
      expect(screen.getByDisplayValue("First")).toBeDefined();
    });
  });
});
