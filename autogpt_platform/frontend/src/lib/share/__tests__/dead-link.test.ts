import { describe, expect, test } from "vitest";

import { isDeadLinkError, retryUnlessDeadLink } from "../dead-link";

describe("isDeadLinkError", () => {
  test.each([404, 422])("is true for a %i", (status) => {
    expect(isDeadLinkError({ status })).toBe(true);
  });

  test.each([
    { failure: "a 500", error: { status: 500 } },
    { failure: "a 502 from the proxy", error: { status: 502 } },
    { failure: "a 429", error: { status: 429 } },
    { failure: "a network error", error: new TypeError("Failed to fetch") },
    { failure: "a status that isn't a number", error: { status: "404" } },
    { failure: "no error at all", error: null },
  ])("is false for $failure", ({ error }) => {
    expect(isDeadLinkError(error)).toBe(false);
  });
});

describe("retryUnlessDeadLink", () => {
  test.each([404, 422])("never retries a %i", (status) => {
    expect(retryUnlessDeadLink(0, { status })).toBe(false);
  });

  test("retries any other failure three times", () => {
    const failureCounts = [0, 1, 2, 3];

    expect(
      failureCounts.map((count) => retryUnlessDeadLink(count, { status: 500 })),
    ).toEqual([true, true, true, false]);
  });
});
