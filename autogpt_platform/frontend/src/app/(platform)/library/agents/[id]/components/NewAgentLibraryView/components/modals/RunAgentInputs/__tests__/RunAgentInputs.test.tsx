import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { BlockIOSubSchema } from "@/lib/autogpt-server-api/types";
import { RunAgentInputs } from "../RunAgentInputs";

function typeInto(schema: BlockIOSubSchema, value: string, current?: number) {
  const onChange = vi.fn();
  render(
    <RunAgentInputs schema={schema} value={current} onChange={onChange} />,
  );
  fireEvent.change(screen.getByRole("spinbutton"), { target: { value } });
  return onChange;
}

const integerSchema = { title: "N", type: "integer" } as BlockIOSubSchema;
const numberSchema = { title: "X", type: "number" } as BlockIOSubSchema;

describe("RunAgentInputs number input", () => {
  it("sends an integer beyond 2^53 as its exact digit string", () => {
    const onChange = typeInto(integerSchema, "9007199254740993");
    expect(onChange).toHaveBeenLastCalledWith("9007199254740993");
  });

  it("does the same for an optional integer", () => {
    const schema = {
      title: "N",
      anyOf: [{ type: "integer" }, { type: "null" }],
    } as BlockIOSubSchema;
    const onChange = typeInto(schema, "-9223372036854775807");
    expect(onChange).toHaveBeenLastCalledWith("-9223372036854775807");
  });

  it("sends safe integers as numbers", () => {
    const onChange = typeInto(integerSchema, "42");
    expect(onChange).toHaveBeenLastCalledWith(42);
  });

  it("sends undefined for an empty integer field", () => {
    const onChange = typeInto(integerSchema, "", 5);
    expect(onChange).toHaveBeenLastCalledWith(undefined);
  });

  it("keeps number fields as plain numbers", () => {
    const onChange = typeInto(numberSchema, "3.7");
    expect(onChange).toHaveBeenLastCalledWith(3.7);
  });

  it("shows a stored digit string unchanged", () => {
    render(
      <RunAgentInputs
        schema={integerSchema}
        value="9007199254740993"
        onChange={vi.fn()}
      />,
    );
    expect(screen.getByRole("spinbutton")).toHaveProperty(
      "value",
      "9007199254740993",
    );
  });
});
