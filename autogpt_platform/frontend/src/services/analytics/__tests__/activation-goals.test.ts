import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  trackAgentRunGoal,
  trackScheduleCreatedGoal,
} from "../activation-goals";

const sendDatafastEvent = vi.hoisted(() => vi.fn());

vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

beforeEach(() => {
  sendDatafastEvent.mockReset();
});

describe("activation goals", () => {
  it("reports a human agent run with its surface", () => {
    trackAgentRunGoal({ id: "graph-1", name: "Daily digest" }, "builder");

    expect(sendDatafastEvent).toHaveBeenCalledWith("run_agent", {
      id: "graph-1",
      name: "Daily digest",
      surface: "builder",
    });
  });

  it("reports a new schedule and tolerates a missing agent name", () => {
    trackScheduleCreatedGoal({ id: "graph-2", name: null }, "library");

    expect(sendDatafastEvent).toHaveBeenCalledWith("schedule_agent", {
      id: "graph-2",
      name: "",
      surface: "library",
    });
  });

  it("never lets a failing tag break the action it describes", () => {
    sendDatafastEvent.mockImplementation(() => {
      throw new Error("tag exploded");
    });

    expect(() => trackAgentRunGoal({ id: "graph-3" }, "rerun")).not.toThrow();
  });
});
