import { describe, expect, it } from "vitest";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { formatToolName } from "../../../tools/GenericTool/helpers";
import { toChainRow } from "../helpers";
import { COPILOT_TOOL_CATALOG } from "../toolCatalog";
// Written by the backend from the registries the model is offered
// (`model_facing_tools_fixture_test.py`), so a new tool lands here on its own.
import modelFacingTools from "./modelFacingTools.json";

function rowText(name: string, state: string): string | undefined {
  return toChainRow(
    {
      type: `tool-${name}`,
      toolCallId: `call-${name}`,
      state,
      input: {},
      output: state === "output-available" ? {} : undefined,
      errorText: state === "output-error" ? "boom" : undefined,
    } as unknown as MessagePart,
    0,
  )?.text;
}

describe("tool catalog coverage", () => {
  it("reads the backend's tool list", () => {
    expect(modelFacingTools).toContain("list_routines");
    expect(modelFacingTools.length).toBeGreaterThan(50);
  });

  // The generic fallback turns the raw tool name into a label ("List
  // routines completed"); every tool the model can call needs better.
  it.each(modelFacingTools)("%s has a human label", (name) => {
    const fallback = formatToolName(name);
    expect(rowText(name, "input-available")).not.toBe(`Running ${fallback}…`);
    expect(rowText(name, "output-available")).not.toBe(`${fallback} completed`);
    expect(rowText(name, "output-error")).not.toBe(`${fallback} failed`);
  });

  it.each(modelFacingTools.filter((name) => name.startsWith("list_")))(
    "%s says what an empty lookup found",
    (name) => {
      expect(COPILOT_TOOL_CATALOG[name]?.empty).toBeTruthy();
    },
  );
});
