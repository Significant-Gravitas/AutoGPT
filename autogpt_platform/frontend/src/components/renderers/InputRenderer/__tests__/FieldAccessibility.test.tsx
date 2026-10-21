import type { RJSFSchema } from "@rjsf/utils";
import { getTestRegistry } from "@rjsf/core";
import type { CustomNode } from "@/app/(platform)/build/components/FlowEditor/nodes/CustomNode/CustomNode";
import { BlockUIType } from "@/app/(platform)/build/components/types";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { render } from "@/tests/integrations/test-utils";
import { act, fireEvent, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { FormRenderer } from "../FormRenderer";
import Form from "../registry";
import { customValidator } from "../utils/custom-validator";
import { DateTimeInput } from "@/components/atoms/DateTimeInput/DateTimeInput";
import {
  MoveDownButton,
  MoveUpButton,
} from "../base/standard/buttons/IconButton";

function node(id: string, error: string): CustomNode {
  return {
    id,
    type: "custom",
    position: { x: 0, y: 0 },
    data: {
      title: id,
      description: "",
      hardcodedValues: {},
      inputSchema: {},
      outputSchema: {},
      uiType: BlockUIType.STANDARD,
      block_id: "test-block",
      costs: [],
      categories: [],
      errors: { field: error },
    },
  };
}

function form(
  id: string,
  property: RJSFSchema,
  onChange = vi.fn(),
  value?: unknown,
) {
  return (
    <FormRenderer
      jsonSchema={{
        type: "object",
        properties: {
          field: {
            ...property,
            title: "Field",
            description: "Helpful description",
          },
        },
      }}
      handleChange={onChange}
      uiSchema={{
        field: { "ui:options": { orderable: true, copyable: true } },
      }}
      initialValues={
        value !== undefined
          ? { field: value }
          : property.type === "object"
            ? { field: {} }
            : property.type === "array"
              ? { field: [] }
              : {}
      }
      formContext={{
        nodeId: id,
        uiType: BlockUIType.STANDARD,
        showHandles: false,
        size: "small",
      }}
    />
  );
}

afterEach(() => {
  useNodeStore.setState({ nodes: [], nodeAdvancedStates: {} });
});

describe("field accessibility", () => {
  it.each([
    ["text", { type: "string" }],
    ["number", { type: "number" }],
    ["boolean", { type: "boolean" }],
    ["select", { type: "string", enum: ["one", "two"] }],
    ["date", { type: "string", format: "date" }],
    ["datetime", { type: "string", format: "date-time" }],
    ["time", { type: "string", format: "time" }],
    [
      "multiselect",
      {
        type: "array",
        uniqueItems: true,
        items: { type: "string", enum: ["one", "two"] },
      },
    ],
  ] satisfies [string, RJSFSchema][])(
    "associates the visible label with the %s control",
    (_name, schema) => {
      render(form("one", schema));
      const control = screen.getByLabelText("Field");
      const labelId = control.getAttribute("aria-labelledby");
      expect(labelId).toBeTruthy();
      expect(document.getElementById(labelId!)?.textContent).toBe("Field");
      const descriptionIds =
        control.getAttribute("aria-describedby")?.split(" ") ?? [];
      expect(descriptionIds.length).toBeGreaterThan(0);
      expect(descriptionIds.every((id) => document.getElementById(id))).toBe(
        true,
      );
    },
  );

  it("keeps repeated node IDs unique and announces only each node's own error", () => {
    useNodeStore.setState({
      nodes: [node("one", "First error"), node("two", "Second error")],
    });
    const { container } = render(
      <>
        {form("one", { type: "string" })}
        {form("two", { type: "string" })}
      </>,
    );
    const ids = Array.from(container.querySelectorAll("[id]")).map(
      (element) => element.id,
    );
    expect(new Set(ids).size).toBe(ids.length);
    const controls = screen.getAllByLabelText("Field");
    for (const [index, control] of controls.entries()) {
      const description = control
        .getAttribute("aria-describedby")
        ?.split(" ")
        .map((id) => document.getElementById(id)?.textContent)
        .join(" ");
      expect(description).toContain(
        index === 0 ? "First error" : "Second error",
      );
      expect(description).not.toContain(
        index === 0 ? "Second error" : "First error",
      );
    }
    act(() => {
      useNodeStore.setState({
        nodes: [node("one", ""), node("two", "New error")],
      });
    });
    const errorId = controls[1]
      .getAttribute("aria-describedby")!
      .split(" ")
      .find(
        (id) =>
          document.getElementById(id)?.getAttribute("aria-live") === "polite",
      );
    expect(document.getElementById(errorId!)?.textContent).toBe("New error");
  });

  it("does not put the DOM namespace into stored field names", () => {
    const onChange = vi.fn();
    render(form("one", { type: "string" }, onChange));
    fireEvent.change(screen.getByLabelText("Field"), {
      target: { value: "saved value" },
    });
    expect(onChange.mock.calls.at(-1)?.[0].formData).toEqual({
      field: "saved value",
    });
  });
});

it.each([
  ["anyOf", { anyOf: [{ type: "string" }, { type: "integer" }] }],
  [
    "oneOf",
    {
      oneOf: [
        { type: "string", title: "Text" },
        { type: "number", title: "Number" },
      ],
    },
  ],
  ["array", { type: "array", items: { type: "string" } }],
  ["dictionary", { type: "object", additionalProperties: { type: "string" } }],
] satisfies [string, RJSFSchema][])(
  "keeps IDs unique across repeated %s fields",
  (_kind, schema) => {
    const { container } = render(
      <>
        {form("one", schema)}
        {form("two", schema)}
      </>,
    );
    const ids = Array.from(container.querySelectorAll("[id]")).map(
      (element) => element.id,
    );
    expect(ids.filter((id, index) => ids.indexOf(id) !== index)).toEqual([]);
  },
);

it("labels nested JSON and links its own validation error", () => {
  const jsonNode = node("one", "");
  jsonNode.data.errors = { "field.payload": "Invalid payload" };
  useNodeStore.setState({ nodes: [jsonNode] });
  render(
    form("one", {
      type: "object",
      properties: {
        payload: {
          type: "object",
          title: "Payload",
          description: "Structured payload",
        },
      },
    }),
  );
  const input = screen.getByLabelText("payload");
  const described = input
    .getAttribute("aria-describedby")!
    .split(" ")
    .map((id) => document.getElementById(id)?.textContent)
    .join(" ");
  expect(described).toContain("Invalid payload");
  expect(
    screen.getByRole("button", { name: "Expand JSON input" }),
  ).not.toBeNull();
});

it("labels the native time control inside a datetime popover", () => {
  render(
    <DateTimeInput id="meeting" label="Meeting" value="2026-09-22T09:00:00" />,
  );
  fireEvent.click(screen.getByRole("button", { name: /meeting/i }));
  expect(screen.getByLabelText("Time").getAttribute("type")).toBe("time");
});

it("describes JSON fields once when the schema retains its description", () => {
  render(
    <Form
      schema={{
        type: "object",
        properties: {
          payload: {
            type: "object",
            title: "Payload",
            description: "Structured payload",
          },
        },
      }}
      uiSchema={{ payload: { "ui:field": "custom/json_text_field" } }}
      validator={customValidator}
      formContext={{
        nodeId: "one",
        domIdPrefix: "description-",
        showHandles: false,
        uiType: BlockUIType.STANDARD,
      }}
    />,
  );
  const described = screen
    .getByLabelText("Payload")
    .getAttribute("aria-describedby")!
    .split(" ")
    .map((id) => document.getElementById(id)?.textContent)
    .join(" ");
  expect(described.match(/Structured payload/g)).toHaveLength(1);
});

it("names array actions for what they do and preserves their behavior", () => {
  const onChange = vi.fn();
  render(
    form("one", { type: "array", items: { type: "string" } }, onChange, [
      "first",
      "second",
    ]),
  );
  fireEvent.click(screen.getAllByRole("button", { name: /^copy$/i })[0]);
  expect(onChange.mock.calls.at(-1)?.[0].formData.field).toEqual([
    "first",
    "first",
    "second",
  ]);
  fireEvent.click(screen.getAllByRole("button", { name: /^remove$/i })[0]);
  expect(onChange.mock.calls.at(-1)?.[0].formData.field).toEqual([
    "first",
    "second",
  ]);
});

it.each([
  ["Move up", MoveUpButton],
  ["Move down", MoveDownButton],
])("uses the same visible and accessible %s label", (label, Component) => {
  const onClick = vi.fn();
  render(<Component registry={getTestRegistry({})} onClick={onClick} />);
  const button = screen.getByRole("button", { name: label });
  expect(button.textContent?.trim()).toBe(label);
  fireEvent.click(button);
  expect(onClick).toHaveBeenCalledOnce();
});

it("keeps prefix-sharing and dollar keys independent of DOM IDs", () => {
  const onChange = vi.fn();
  const { container } = render(
    form(
      "one",
      { type: "object", additionalProperties: { type: "string" } },
      onChange,
      { "cost$&": "first", "cost$&-test": "second" },
    ),
  );
  const ids = Array.from(container.querySelectorAll("[id]")).map(
    (element) => element.id,
  );
  expect(ids.filter((id, index) => ids.indexOf(id) !== index)).toEqual([]);
  expect(screen.getByLabelText("cost$& key")).not.toBeNull();
  expect(screen.getByLabelText("cost$&-test key")).not.toBeNull();
  fireEvent.change(screen.getByDisplayValue("first"), {
    target: { value: "changed" },
  });
  expect(onChange.mock.calls.at(-1)?.[0].formData.field).toEqual({
    "cost$&": "changed",
    "cost$&-test": "second",
  });
});

it("names the expanded input and clipboard action", async () => {
  const descriptor = Object.getOwnPropertyDescriptor(navigator, "clipboard");
  const writeText = vi.fn().mockResolvedValue(undefined);
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText },
  });
  try {
    render(form("one", { type: "string" }, vi.fn(), "copy this"));
    fireEvent.click(screen.getByRole("button", { name: "Expand input" }));
    expect(screen.getByRole("textbox", { name: "Field" })).not.toBeNull();
    await act(async () => {
      fireEvent.click(
        screen.getByRole("button", { name: "Copy to clipboard" }),
      );
    });
    expect(writeText).toHaveBeenCalledWith("copy this");
    expect(
      screen.getByRole("button", { name: "Copied to clipboard" }),
    ).not.toBeNull();
  } finally {
    if (descriptor) Object.defineProperty(navigator, "clipboard", descriptor);
    else Reflect.deleteProperty(navigator, "clipboard");
  }
});
