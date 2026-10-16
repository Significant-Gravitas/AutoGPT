import type { UIDataTypes, UIMessage, UITools } from "ai";
import { expect, test } from "vitest";
import { getHeldOutcomes } from "../heldCallRows";

function lateResult(
  metadata: Record<string, unknown>,
  body: string,
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id: "m1",
    role: "user",
    metadata,
    parts: [
      {
        type: "text",
        text: `<held_call_result tool="create_folder" tool_call_id="call-7" review_id="r">\n${body}\n</held_call_result>`,
      },
    ],
  };
}

test("a late result is keyed by the original call and carries its outcome and output", () => {
  const outcomes = getHeldOutcomes([
    lateResult(
      { held_call: { tool_call_id: "call-7", outcome: "approved" } },
      '{"type":"folder_created","name":"Q3"}',
    ),
  ]);
  expect(outcomes.get("call-7")).toEqual({
    outcome: "approved",
    output: { type: "folder_created", name: "Q3" },
  });
});

test("a row from before outcomes were recorded is read from its text", () => {
  const refused = getHeldOutcomes([
    lateResult(
      { held_call: { tool_call_id: "call-7" } },
      "Nothing ran: declined",
    ),
  ]);
  expect(refused.get("call-7")?.outcome).toBe("closed");
});

test("an unknown outcome reads as approved, and plain text stays text", () => {
  const outcomes = getHeldOutcomes([
    lateResult({ held_call: { tool_call_id: "call-7" } }, "posted"),
  ]);
  expect(outcomes.get("call-7")).toEqual({
    outcome: "approved",
    output: "posted",
  });
});

test("rows that are not late results are ignored", () => {
  expect(
    getHeldOutcomes([lateResult({ held_calls_answered: true }, "x")]).size,
  ).toBe(0);
});

test("an output that contains the closing tag is kept whole", () => {
  const body = 'wrote "</held_call_result>" into notes.md';
  const outcomes = getHeldOutcomes([
    lateResult(
      { held_call: { tool_call_id: "call-7", outcome: "approved" } },
      body,
    ),
  ]);
  expect(outcomes.get("call-7")?.output).toBe(body);
});

function heldCall(
  toolCallId: string,
  reviewId: string,
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id: `a-${toolCallId}`,
    role: "assistant",
    parts: [
      {
        type: "tool-bash_exec",
        toolCallId,
        state: "output-available",
        input: {},
        output: JSON.stringify({
          type: "approval_required",
          review_id: reviewId,
        }),
      },
    ],
  };
}

const standIn = (reviewId: string, outcome: string) =>
  lateResult(
    {
      held_call: {
        tool_call_id: "sdk-365200f8ce9e",
        review_id: reviewId,
        outcome,
      },
    },
    "done",
  );

test("a result under the SDK engine's stand-in id settles the model's held call by review id", () => {
  const outcomes = getHeldOutcomes([
    heldCall("toolu_01", "gate:abc"),
    standIn("gate:abc", "approved"),
  ]);
  expect(outcomes.get("toolu_01")?.outcome).toBe("approved");
  expect(outcomes.get("sdk-365200f8ce9e")?.outcome).toBe("approved");
});

test("a repeat of the same call waits for its own result, not the first one's", () => {
  const outcomes = getHeldOutcomes([
    heldCall("toolu_01", "gate:abc"),
    standIn("gate:abc", "rejected"),
    heldCall("toolu_02", "gate:abc"),
  ]);
  expect(outcomes.get("toolu_01")?.outcome).toBe("rejected");
  expect(outcomes.has("toolu_02")).toBe(false);
});
