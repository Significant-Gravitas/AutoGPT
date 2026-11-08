import type { RJSFSchema } from "@rjsf/utils";
import { screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BlockUIType } from "@/app/(platform)/build/components/types";
import { useEdgeStore } from "@/app/(platform)/build/stores/edgeStore";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { render } from "@/tests/integrations/test-utils";
import { FormRenderer } from "../FormRenderer";

function freeze(value: unknown) {
  if (value && typeof value === "object") {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
}

function renderInput(property: RJSFSchema, value: unknown) {
  const schema: RJSFSchema = {
    type: "object",
    properties: { payload: { title: "Payload", ...property } },
  };
  const original = structuredClone(schema);
  freeze(schema);
  const onChange = vi.fn();
  render(
    <FormRenderer
      jsonSchema={schema}
      initialValues={{ payload: value }}
      handleChange={onChange}
      uiSchema={{}}
      formContext={{
        nodeId: "anyof-node",
        uiType: BlockUIType.STANDARD,
        showHandles: false,
        size: "small",
      }}
    />,
  );
  expect(schema).toEqual(original);
  return onChange;
}

beforeEach(() => {
  useEdgeStore.setState({ edges: [] });
  useNodeStore.setState({ nodes: [], nodeAdvancedStates: {} });
});

afterEach(() => vi.restoreAllMocks());

describe("anyOf through the real InputRenderer registry", () => {
  it.each([
    ["untyped", {}],
    [
      "explicit union",
      {
        anyOf: [
          { type: "string" },
          { type: "object", additionalProperties: true },
          { type: "null" },
        ],
      },
    ],
    [
      "nullable object",
      {
        anyOf: [
          { type: "null" },
          { type: "object", additionalProperties: true },
        ],
      },
    ],
  ] satisfies [string, RJSFSchema][])(
    "renders an existing empty %s object without invalid anyOf compilation",
    (_name, property) => {
      const warn = vi.spyOn(console, "warn");
      const onChange = renderInput(property, {});

      expect(screen.getByRole("button", { name: /^add item$/i })).toBeDefined();
      expect(screen.queryByRole("textbox")).toBeNull();
      expect(warn.mock.calls.flat().map(String).join("\n")).not.toContain(
        "schema is invalid",
      );
      expect(onChange).not.toHaveBeenCalled();
    },
  );

  it.each([null, undefined])(
    "keeps the default string choice for %s untyped data",
    (value) => {
      const onChange = renderInput({}, value);
      expect(screen.getByRole("combobox").textContent).toContain("string");
      expect(screen.getByRole("textbox")).toBeDefined();
      expect(onChange).not.toHaveBeenCalled();
    },
  );

  it("keeps closed object constraints when selecting an empty object", () => {
    renderInput(
      {
        anyOf: [
          { type: "string" },
          { type: "object", properties: {}, additionalProperties: false },
          { type: "null" },
        ],
      },
      {},
    );
    expect(screen.getByRole("combobox").textContent).toContain("object");
    expect(screen.queryByRole("button", { name: /^add item$/i })).toBeNull();
  });

  it("keeps nonempty object fields and required labels", () => {
    renderInput(
      {
        anyOf: [
          { type: "string" },
          {
            type: "object",
            properties: { name: { type: "string", title: "Name" } },
            required: ["name"],
          },
          { type: "null" },
        ],
      },
      { name: "Existing" },
    );
    expect(screen.getByDisplayValue("Existing")).toBeDefined();
    expect(screen.getByRole("combobox").textContent).toContain("object");
    expect(screen.getByText("Name").parentElement?.textContent).toContain("*");
  });

  it("keeps declared boolean objects in the existing multiselect renderer", () => {
    renderInput(
      {
        anyOf: [
          { type: "string" },
          {
            type: "object",
            properties: { enabled: { type: "boolean", title: "Enabled" } },
          },
          { type: "null" },
        ],
      },
      { enabled: true },
    );
    expect(
      screen.getByRole("combobox", { name: "Payload — variant" }).textContent,
    ).toContain("object");
    expect(screen.getByText("enabled")).toBeDefined();
    expect(screen.queryByRole("button", { name: /^add item$/i })).toBeNull();
  });

  it("characterizes the inherited null-member failure before anyOf field rendering", () => {
    vi.spyOn(console, "error").mockImplementation(() => {});
    const property: RJSFSchema = JSON.parse(
      '{"anyOf":[{"type":"string"},null]}',
    );
    expect(() => renderInput(property, {})).toThrow(/properties/);
  });
});
