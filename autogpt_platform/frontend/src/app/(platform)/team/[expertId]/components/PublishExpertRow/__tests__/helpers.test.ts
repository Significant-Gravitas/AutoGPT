import { describe, expect, test } from "vitest";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { getPublishErrorMessage } from "../helpers";

function apiError(status: number, response: unknown, message = "") {
  const error = new ApiError(message, status, response);
  return error;
}

describe("getPublishErrorMessage", () => {
  test("names the agents the publish route refused", () => {
    const error = apiError(400, {
      detail: { code: "unpublished_workflows", workflows: ["Tick", "Tock"] },
    });

    expect(getPublishErrorMessage(error, "Frankie")).toBe(
      "Publish these agents to the marketplace first: Tick, Tock",
    );
  });

  test("falls back when the body carries the code but names nothing", () => {
    // An empty list is still truthy; naming no agents at all reads worse than
    // the generic failure.
    const error = apiError(400, {
      detail: { code: "unpublished_workflows", workflows: [] },
    });

    expect(getPublishErrorMessage(error, "Frankie")).toBe(
      "Couldn't publish Frankie",
    );
  });

  test("falls back when the list is missing or not a list", () => {
    for (const workflows of [undefined, "Tick", 3]) {
      const error = apiError(400, {
        detail: { code: "unpublished_workflows", workflows },
      });
      expect(getPublishErrorMessage(error, "Frankie")).toBe(
        "Couldn't publish Frankie",
      );
    }
  });

  test("drops non-string entries but keeps the nameable ones", () => {
    const error = apiError(400, {
      detail: { code: "unpublished_workflows", workflows: ["Tick", 7, null] },
    });

    expect(getPublishErrorMessage(error, "Frankie")).toBe(
      "Publish these agents to the marketplace first: Tick",
    );
  });

  test("explains a 403 as an admin-only action", () => {
    expect(getPublishErrorMessage(apiError(403, {}), "Frankie")).toBe(
      "Only admins can publish",
    );
  });

  test("uses a plain error's message, then the fallback", () => {
    expect(getPublishErrorMessage(new Error("network down"), "Frankie")).toBe(
      "network down",
    );
    expect(getPublishErrorMessage(new Error(""), "Frankie")).toBe(
      "Couldn't publish Frankie",
    );
    expect(getPublishErrorMessage("nope", "Frankie")).toBe(
      "Couldn't publish Frankie",
    );
  });
});
