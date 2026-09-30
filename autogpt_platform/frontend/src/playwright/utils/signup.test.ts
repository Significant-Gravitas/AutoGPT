import { describe, expect, it } from "vitest";

import { isAuthenticatedAppURL } from "./signup";

describe("isAuthenticatedAppURL", () => {
  it.each(["/home", "/copilot", "/library"])(
    "accepts %s as an authenticated app route",
    (path) => {
      expect(isAuthenticatedAppURL(`http://localhost${path}`)).toBe(true);
    },
  );

  it("leaves marketplace verification on its own path", () => {
    expect(isAuthenticatedAppURL("http://localhost/marketplace")).toBe(false);
  });

  it("accepts nested authenticated app routes", () => {
    expect(isAuthenticatedAppURL("/home/session/123")).toBe(true);
  });

  it("does not match authenticated routes in query parameters", () => {
    expect(isAuthenticatedAppURL("/marketplace?returnTo=/home")).toBe(false);
  });
});
