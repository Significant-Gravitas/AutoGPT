import { afterEach, describe, expect, it, vi } from "vitest";
import {
  forgetHeldFollowUp,
  rememberHeldFollowUp,
  takeHeldFollowUps,
} from "../heldFollowUps";

afterEach(() => {
  window.sessionStorage.clear();
});

describe("heldFollowUps", () => {
  it("hands back what was remembered for a chat, once", () => {
    rememberHeldFollowUp("s1", "first");
    rememberHeldFollowUp("s1", "second");
    rememberHeldFollowUp("s2", "elsewhere");

    expect(takeHeldFollowUps("s1")).toEqual(["first", "second"]);
    expect(takeHeldFollowUps("s1")).toEqual([]);
    expect(takeHeldFollowUps("s2")).toEqual(["elsewhere"]);
  });

  it("forgets only the one copy that went out", () => {
    rememberHeldFollowUp("s1", "same");
    rememberHeldFollowUp("s1", "same");
    forgetHeldFollowUp("s1", "same");
    expect(takeHeldFollowUps("s1")).toEqual(["same"]);
  });

  it("drops the key once nothing is held", () => {
    rememberHeldFollowUp("s1", "only");
    forgetHeldFollowUp("s1", "only");
    expect(window.sessionStorage.length).toBe(0);
  });

  it("keeps going when storage refuses the write", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const real = window.sessionStorage;
    const full = {
      getItem: () => null,
      setItem: () => {
        throw new DOMException("full", "QuotaExceededError");
      },
      removeItem: () => {
        throw new DOMException("blocked", "SecurityError");
      },
    };
    Object.defineProperty(window, "sessionStorage", {
      value: full,
      configurable: true,
    });
    try {
      expect(() => rememberHeldFollowUp("s1", "text")).not.toThrow();
      expect(() => takeHeldFollowUps("s1")).not.toThrow();
      expect(warn).toHaveBeenCalledTimes(2);
    } finally {
      Object.defineProperty(window, "sessionStorage", {
        value: real,
        configurable: true,
      });
      warn.mockRestore();
    }
  });

  it("ignores a corrupt entry instead of throwing", () => {
    window.sessionStorage.setItem("copilot-held-follow-ups:s1", "{not json");
    expect(takeHeldFollowUps("s1")).toEqual([]);
    window.sessionStorage.setItem(
      "copilot-held-follow-ups:s1",
      JSON.stringify([1, "text", null]),
    );
    expect(takeHeldFollowUps("s1")).toEqual(["text"]);
  });
});
