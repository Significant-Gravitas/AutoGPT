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
  it("reports a human agent run with its surface and no workflow name", () => {
    trackAgentRunGoal("graph-1", "builder");

    expect(sendDatafastEvent).toHaveBeenCalledWith("run_agent", {
      id: "graph-1",
      surface: "builder",
    });
  });

  it("reports a new schedule with its surface and no workflow name", () => {
    trackScheduleCreatedGoal("graph-2", "library");

    expect(sendDatafastEvent).toHaveBeenCalledWith("schedule_agent", {
      id: "graph-2",
      surface: "library",
    });
  });

  it("never lets a failing tag break the action it describes", () => {
    sendDatafastEvent.mockImplementation(() => {
      throw new Error("tag exploded");
    });

    expect(() => trackAgentRunGoal("graph-3", "rerun")).not.toThrow();
  });
});
