import { describe, expect, it } from "vitest";
import { convertChatSessionMessagesToUiMessages } from "../../helpers/convertChatSessionToUiMessages";
import { loadRecordedTurn, type Row } from "./backend-sim";

// The SDK engine used to add a tool call that follows a tool result to the
// earlier assistant row; it now opens a row for it, as the baseline always
// did. A reload must draw the same bubble from either shape.
describe("a tool call after a tool result, on reload", () => {
  it("renders the same whether it has its own row or joins the earlier one", () => {
    const rows = loadRecordedTurn("sdk-consecutive-tools-turn").rows;
    const [prompt, firstCall, firstResult, secondCall, ...rest] = rows;
    const joined: Row[] = [
      prompt,
      {
        ...firstCall,
        tool_calls: [
          ...(firstCall.tool_calls as unknown[]),
          ...(secondCall.tool_calls as unknown[]),
        ],
      },
      firstResult,
      ...rest,
    ];

    expect(render(rows)).toEqual(render(joined));
    expect(render(rows).map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
  });
});

function render(rows: Row[]) {
  return convertChatSessionMessagesToUiMessages("drift-session", rows, {
    isComplete: true,
  }).messages.map(({ role, parts }) => ({ role, parts }));
}
