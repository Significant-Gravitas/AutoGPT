import { describe, it, expect } from "vitest";
import {
  isEmptyExecutionUpdate,
  isPollingStatus,
  EMPTY_EXECUTION_UPDATES_THRESHOLD,
} from "./executionPollingWatchdog";

describe("isEmptyExecutionUpdate", () => {
  it("returns false for null or non-object", () => {
    expect(isEmptyExecutionUpdate(null)).toBe(false);
    expect(isEmptyExecutionUpdate(undefined)).toBe(false);
    expect(isEmptyExecutionUpdate("")).toBe(false);
    expect(isEmptyExecutionUpdate(42)).toBe(false);
    expect(isEmptyExecutionUpdate([])).toBe(false);
  });

  it("returns false when HTTP status is not 200", () => {
    expect(
      isEmptyExecutionUpdate({
        status: 404,
        data: { status: "RUNNING", node_executions: [] },
      }),
    ).toBe(false);
    expect(
      isEmptyExecutionUpdate({
        status: 500,
        data: { status: "RUNNING", node_executions: [] },
      }),
    ).toBe(false);
  });

  it("returns false for malformed payload (no data)", () => {
    expect(isEmptyExecutionUpdate({ status: 200 })).toBe(false);
    expect(isEmptyExecutionUpdate({ status: 200, data: null })).toBe(false);
    expect(isEmptyExecutionUpdate({ status: 200, data: [] })).toBe(false);
  });

  it("returns false for terminal status", () => {
    const withStatus = (s: string) => ({
      status: 200,
      data: { status: s, node_executions: [] },
    });
    expect(isEmptyExecutionUpdate(withStatus("COMPLETED"))).toBe(false);
    expect(isEmptyExecutionUpdate(withStatus("FAILED"))).toBe(false);
    expect(isEmptyExecutionUpdate(withStatus("TERMINATED"))).toBe(false);
  });

  it("returns true for polling status with empty node_executions", () => {
    const withStatus = (s: string) => ({
      status: 200,
      data: { status: s, node_executions: [] },
    });
    expect(isEmptyExecutionUpdate(withStatus("RUNNING"))).toBe(true);
    expect(isEmptyExecutionUpdate(withStatus("QUEUED"))).toBe(true);
    expect(isEmptyExecutionUpdate(withStatus("INCOMPLETE"))).toBe(true);
    expect(isEmptyExecutionUpdate(withStatus("REVIEW"))).toBe(true);
  });

  it("returns false for polling status with non-empty node_executions", () => {
    expect(
      isEmptyExecutionUpdate({
        status: 200,
        data: { status: "RUNNING", node_executions: [{ id: "1" }] },
      }),
    ).toBe(false);
  });

  it("returns false when node_executions is not an array", () => {
    expect(
      isEmptyExecutionUpdate({
        status: 200,
        data: { status: "RUNNING", node_executions: null },
      }),
    ).toBe(false);
    expect(
      isEmptyExecutionUpdate({
        status: 200,
        data: { status: "RUNNING", node_executions: {} },
      }),
    ).toBe(false);
  });

  it("returns false for unknown status even with empty node_executions", () => {
    expect(
      isEmptyExecutionUpdate({
        status: 200,
        data: { status: "UNKNOWN", node_executions: [] },
      }),
    ).toBe(false);
  });
});

describe("isPollingStatus", () => {
  it("returns false for undefined or empty", () => {
    expect(isPollingStatus(undefined)).toBe(false);
    expect(isPollingStatus("")).toBe(false);
  });

  it("returns true for polling statuses", () => {
    expect(isPollingStatus("RUNNING")).toBe(true);
    expect(isPollingStatus("QUEUED")).toBe(true);
    expect(isPollingStatus("INCOMPLETE")).toBe(true);
    expect(isPollingStatus("REVIEW")).toBe(true);
  });

  it("returns false for terminal statuses", () => {
    expect(isPollingStatus("COMPLETED")).toBe(false);
    expect(isPollingStatus("FAILED")).toBe(false);
    expect(isPollingStatus("TERMINATED")).toBe(false);
  });
});

describe("EMPTY_EXECUTION_UPDATES_THRESHOLD", () => {
  it("is a positive number", () => {
    expect(EMPTY_EXECUTION_UPDATES_THRESHOLD).toBe(40);
    expect(EMPTY_EXECUTION_UPDATES_THRESHOLD).toBeGreaterThan(0);
  });
});
