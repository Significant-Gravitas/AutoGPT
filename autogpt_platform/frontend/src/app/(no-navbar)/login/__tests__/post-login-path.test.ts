import { describe, expect, test } from "vitest";
import { getPostLoginPath } from "../helpers";

describe("getPostLoginPath", () => {
  test("a closed account goes to the keep-account page, even with ?next=", () => {
    expect(getPostLoginPath("/copilot", "/account-closed")).toBe(
      "/account-closed",
    );
    expect(getPostLoginPath("/marketplace", "/account-closed")).toBe(
      "/account-closed",
    );
  });

  test("an open account honours ?next= first, then the server's choice", () => {
    expect(getPostLoginPath("/library", "/home")).toBe("/library");
    expect(getPostLoginPath(null, "/onboarding")).toBe("/onboarding");
    expect(getPostLoginPath(null, undefined)).toBe("/");
  });
});
