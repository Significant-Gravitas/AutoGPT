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
    expect(getHostFromUrl("api.example.com:8443")).toBe("api.example.com:8443");
  });

  it("keeps a non-default port", () => {
    expect(getHostFromUrl("https://api.example.com:8443/v1")).toBe(
      "api.example.com:8443",
    );
    expect(getHostFromUrl("http://localhost:8080/api")).toBe("localhost:8080");
    expect(getHostFromUrl("HTTPS://api.example.com:8443/v1")).toBe(
      "api.example.com:8443",
    );
  });

  it("drops a port that is the scheme's default", () => {
    expect(getHostFromUrl("https://x.com:443/")).toBe("x.com");
    expect(getHostFromUrl("http://x.com:80/")).toBe("x.com");
  });

  it("returns null for a string it cannot parse", () => {
    expect(getHostFromUrl("")).toBeNull();
    expect(getHostFromUrl("http://")).toBeNull();
  });
});
