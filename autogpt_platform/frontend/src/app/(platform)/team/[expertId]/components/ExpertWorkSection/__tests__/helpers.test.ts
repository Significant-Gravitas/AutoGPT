import { describe, expect, it } from "vitest";
import { getRunStatusInfo } from "../helpers";

describe("getRunStatusInfo", () => {
  it.each([
    ["COMPLETED", "Completed"],
    ["FAILED", "Failed"],
    ["RUNNING", "Running"],
    ["QUEUED", "Queued"],
    ["REVIEW", "Waiting for review"],
    ["TERMINATED", "Stopped"],
    ["INCOMPLETE", "Incomplete"],
  ])("labels %s as %s", (status, label) => {
    expect(getRunStatusInfo(status).label).toBe(label);
  });

  it("only COMPLETED reads as completed", () => {
    const nonTerminal = [
      "FAILED",
      "RUNNING",
      "QUEUED",
      "REVIEW",
      "TERMINATED",
      "INCOMPLETE",
    ];
    for (const status of nonTerminal) {
      expect(getRunStatusInfo(status).label).not.toBe("Completed");
    }
  });

  it("is case-insensitive and falls back to the raw value", () => {
    expect(getRunStatusInfo("completed").label).toBe("Completed");
    expect(getRunStatusInfo("SOMETHING_NEW").label).toBe("SOMETHING_NEW");
  });
});
