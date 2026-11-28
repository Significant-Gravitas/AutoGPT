// @vitest-environment node
import { describe, expect, it } from "vitest";
import { assertLoaded, parseConcurrency } from "../ds/canary-check";

describe("parseConcurrency", () => {
  it("defaults to 4 workers and takes a whole number of 1 or more", () => {
    expect(parseConcurrency(undefined)).toBe(4);
    expect(parseConcurrency("1")).toBe(1);
    expect(parseConcurrency("8")).toBe(8);
  });

  it("rejects a value that would start no worker or a fraction of one", () => {
    for (const value of ["", "0", "-2", "1.5", "two", "Infinity", "NaN"])
      expect(parseConcurrency(value)).toBeNull();
  });
});

describe("assertLoaded", () => {
  const response = (status: number) => ({
    ok: () => status >= 200 && status < 300,
    status: () => status,
  });

  it("fails a navigation that answered with an HTTP error", () => {
    expect(() => assertLoaded(response(500), "http://app/login")).toThrow(
      "http://app/login answered HTTP 500",
    );
    expect(() => assertLoaded(response(404), "http://app/x")).toThrow(
      /HTTP 404/,
    );
  });

  it("passes a page that loaded, and a navigation with no response", () => {
    expect(() => assertLoaded(response(200), "http://app/login")).not.toThrow();
    expect(() => assertLoaded(null, "http://app/login")).not.toThrow();
  });
});
