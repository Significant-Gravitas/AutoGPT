import { describe, expect, it } from "vitest";

import { getHostFromUrl } from "../url";

describe("getHostFromUrl", () => {
  it("reads the host from a lowercase scheme", () => {
    expect(getHostFromUrl("https://api.example.com/v1")).toBe(
      "api.example.com",
    );
    expect(getHostFromUrl("http://api.example.com")).toBe("api.example.com");
  });

  it("reads the host from an uppercase or mixed-case scheme", () => {
    expect(getHostFromUrl("HTTPS://api.example.com/v1")).toBe(
      "api.example.com",
    );
    expect(getHostFromUrl("HTTP://api.example.com")).toBe("api.example.com");
    expect(getHostFromUrl("Https://api.example.com/v1")).toBe(
      "api.example.com",
    );
  });

  it("assumes http for a bare host", () => {
    expect(getHostFromUrl("api.example.com/v1")).toBe("api.example.com");
    expect(getHostFromUrl("api.example.com:8443")).toBe("api.example.com");
  });

  it("returns null for a string it cannot parse", () => {
    expect(getHostFromUrl("")).toBeNull();
    expect(getHostFromUrl("http://")).toBeNull();
  });
});
