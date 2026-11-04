import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import { getLatestPlanId } from "../latestPlan";

type Message = UIMessage<unknown, UIDataTypes, UITools>;

function planMessage(id: string, planId: string): Message {
  return {
    id,
    role: "assistant",
    parts: [
      {
        type: "tool-propose_plan",
        toolCallId: `call-${id}`,
        state: "output-available",
        input: {},
        output: {
          type: "task_plan",
          plan: { id: planId, goal: "Launch", steps: [{ title: "Copy" }] },
        },
      },
    ],
  } as unknown as Message;
}

describe("getLatestPlanId", () => {
  it("is the newest proposed plan in the session", () => {
    expect(
      getLatestPlanId([
        planMessage("m1", "plan-1"),
        planMessage("m2", "plan-2"),
      ]),
    ).toBe("plan-2");
  });

  it("is null without a plan", () => {
    expect(getLatestPlanId([])).toBeNull();
  });
});
