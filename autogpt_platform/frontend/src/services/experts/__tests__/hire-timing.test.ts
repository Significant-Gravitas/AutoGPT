import { afterEach, describe, expect, it, vi } from "vitest";
import { markHireStarted, takeHireElapsedMs } from "../hire-timing";

afterEach(() => {
  vi.restoreAllMocks();
  window.sessionStorage.clear();
});

describe("hire timing", () => {
  it("reports the span between the hire click and the finish", () => {
    vi.spyOn(Date, "now").mockReturnValue(1_000);
    markHireStarted("template-maria");

    vi.spyOn(Date, "now").mockReturnValue(4_500);
    expect(takeHireElapsedMs("template-maria")).toBe(3_500);
  });

  it("stores the mark per template", () => {
    markHireStarted("template-maria");

    expect(takeHireElapsedMs("template-max")).toBeNull();
    expect(takeHireElapsedMs("template-maria")).not.toBeNull();
  });

  it("clears the mark so a second finish cannot report a stale span", () => {
    markHireStarted("template-maria");

    expect(takeHireElapsedMs("template-maria")).not.toBeNull();
    expect(takeHireElapsedMs("template-maria")).toBeNull();
  });

  it("returns null when the flow was never started", () => {
    expect(takeHireElapsedMs("template-never")).toBeNull();
  });

  it("ignores a corrupted mark", () => {
    window.sessionStorage.setItem("autogpt:hire-started:template-maria", "abc");

    expect(takeHireElapsedMs("template-maria")).toBeNull();
  });

  it("survives a storage that refuses reads and writes", () => {
    vi.spyOn(window.sessionStorage, "setItem").mockImplementation(() => {
      throw new Error("blocked");
    });
    vi.spyOn(window.sessionStorage, "getItem").mockImplementation(() => {
      throw new Error("blocked");
    });

    expect(() => markHireStarted("template-maria")).not.toThrow();
    expect(takeHireElapsedMs("template-maria")).toBeNull();
  });
});
