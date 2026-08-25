import { describe, expect, it } from "vitest";

import {
  createDemoAccessGate,
  createDemoAccessRateLimiter,
  DEMO_ACCESS_TTL_SECONDS,
} from "./demo-access.js";

describe("demo access gate", () => {
  it("stays disabled when no code is configured", () => {
    const gate = createDemoAccessGate(undefined);

    expect(gate.enabled).toBe(false);
    expect(gate.acceptsCode("")).toBe(false);
    expect(gate.acceptsCookie(undefined)).toBe(false);
  });

  it("exchanges only the configured code for a derived cookie", () => {
    const gate = createDemoAccessGate("correct-horse-battery-staple");

    expect(gate.enabled).toBe(true);
    expect(gate.acceptsCode("correct-horse-battery-staple")).toBe(true);
    expect(gate.acceptsCode("wrong-code-value")).toBe(false);
    expect(gate.cookieValue()).not.toContain("correct-horse-battery-staple");
    expect(gate.acceptsCookie(gate.cookieValue())).toBe(true);
  });

  it("rejects weak configured codes at startup", () => {
    expect(() => createDemoAccessGate("too-short")).toThrow(
      "at least 16 characters",
    );
  });

  it("fails closed when public mode requires a code", () => {
    expect(() => createDemoAccessGate(undefined, { required: true })).toThrow(
      "required in public demo mode",
    );
  });

  it("rejects an expired copied cookie server-side", () => {
    let now = 1_700_000_000_000;
    const gate = createDemoAccessGate("correct-horse-battery-staple", {
      now: () => now,
    });
    const cookie = gate.cookieValue();

    expect(gate.acceptsCookie(cookie)).toBe(true);
    now += DEMO_ACCESS_TTL_SECONDS * 1000;
    expect(gate.acceptsCookie(cookie)).toBe(false);
  });

  it("limits attempts per client and resets after success", () => {
    let now = 1_700_000_000_000;
    const limiter = createDemoAccessRateLimiter(2, 60_000, () => now);

    expect(limiter.consume("client").allowed).toBe(true);
    expect(limiter.consume("client").allowed).toBe(true);
    expect(limiter.consume("client")).toEqual({
      allowed: false,
      retryAfterSeconds: 60,
    });
    limiter.reset("client");
    expect(limiter.consume("client").allowed).toBe(true);
    now += 60_000;
    expect(limiter.consume("client").allowed).toBe(true);
  });

  it("evicts expired clients and fails closed at its memory bound", () => {
    let now = 1_700_000_000_000;
    const limiter = createDemoAccessRateLimiter(2, 60_000, () => now, 2);

    expect(limiter.consume("client-a").allowed).toBe(true);
    expect(limiter.consume("client-b").allowed).toBe(true);
    expect(limiter.consume("client-c")).toEqual({
      allowed: false,
      retryAfterSeconds: 60,
    });
    now += 60_000;
    expect(limiter.consume("client-c").allowed).toBe(true);
  });
});
