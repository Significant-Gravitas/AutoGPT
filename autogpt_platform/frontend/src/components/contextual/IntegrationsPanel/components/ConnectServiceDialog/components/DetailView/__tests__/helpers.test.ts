import { describe, expect, it } from "vitest";

import { getOAuthErrorMessage } from "../helpers";

function apiError(response: unknown, message = "[object Object]") {
  const err = new Error(message) as Error & { response: unknown };
  err.name = "ApiError";
  err.response = response;
  return err;
}

describe("getOAuthErrorMessage", () => {
  it("reads a string detail from response", () => {
    expect(getOAuthErrorMessage(apiError({ detail: "boom" }))).toBe("boom");
  });

  it("joins msg fields from a validation array", () => {
    const err = apiError({
      detail: [{ msg: "Field required" }, { msg: "Too short" }],
    });
    expect(getOAuthErrorMessage(err)).toBe("Field required, Too short");
  });

  it("combines message and hint from a dict detail", () => {
    const err = apiError({
      detail: { message: "not configured", hint: "set keys" },
    });
    expect(getOAuthErrorMessage(err)).toBe("not configured set keys");
  });

  it("returns message alone when the dict has no hint", () => {
    expect(
      getOAuthErrorMessage(apiError({ detail: { message: "nope" } })),
    ).toBe("nope");
  });

  it("falls back to detail on the error itself when there is no response", () => {
    const err = new Error("[object Object]") as Error & { detail: unknown };
    err.detail = "direct detail";
    expect(getOAuthErrorMessage(err)).toBe("direct detail");
  });

  it("uses error.message when no usable detail is present", () => {
    expect(getOAuthErrorMessage(new Error("plain failure"))).toBe(
      "plain failure",
    );
  });

  it("falls back to a generic message when message is the coerced object string", () => {
    expect(getOAuthErrorMessage(apiError({ detail: { code: 1 } }))).toBe(
      "Something went wrong. Please try again.",
    );
  });

  it("falls back to a generic message for non-error values", () => {
    expect(getOAuthErrorMessage("just a string")).toBe(
      "Something went wrong. Please try again.",
    );
    expect(getOAuthErrorMessage(null)).toBe(
      "Something went wrong. Please try again.",
    );
  });
});
