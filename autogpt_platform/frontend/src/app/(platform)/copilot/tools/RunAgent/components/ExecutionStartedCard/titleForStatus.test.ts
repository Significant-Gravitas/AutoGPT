import { describe, expect, it } from "vitest";
import { titleForStatus } from "./ExecutionStartedCard";

describe("titleForStatus", () => {
  it.each([
    ["COMPLETED", "Execution completed"],
    ["FAILED", "Execution failed"],
    ["STOPPED", "Execution stopped"],
    ["TERMINATED", "Execution stopped"],
    ["CANCELLED", "Execution stopped"],
    ["TIMED_OUT", "Execution incomplete"],
    ["INCOMPLETE", "Execution incomplete"],
    ["RUNNING", "Execution running"],
    ["QUEUED", "Execution started"],
    ["", "Execution started"],
  ])("maps %s -> %s", (input, expected) => {
    expect(titleForStatus(input)).toBe(expected);
  });

  it("treats undefined status as just-started", () => {
    expect(titleForStatus(undefined)).toBe("Execution started");
  });

  it("is case-insensitive", () => {
    expect(titleForStatus("completed")).toBe("Execution completed");
    expect(titleForStatus("Failed")).toBe("Execution failed");
  });

  it("falls back to the started label for unknown statuses", () => {
    expect(titleForStatus("WEIRD_CUSTOM_STATUS")).toBe("Execution started");
  });
});
