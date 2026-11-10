import type { CapturedNetworkRequest } from "posthog-js";
import { describe, expect, it } from "vitest";
import { maskCapturedNetworkRequest } from "../posthog-replay";

function request(name: string): CapturedNetworkRequest {
  return {
    name,
    entryType: "resource",
    startTime: 10,
    duration: 5,
    method: "GET",
    status: 200,
  };
}

describe("maskCapturedNetworkRequest", () => {
  it("keeps a request's address and drops its query string", () => {
    expect(
      maskCapturedNetworkRequest(
        request("https://api.example.com/rtc/validate?access_token=abc&x=1"),
      ),
    ).toEqual(request("https://api.example.com/rtc/validate"));
  });

  it("drops a fragment along with the query", () => {
    expect(
      maskCapturedNetworkRequest(request("https://example.com/a?b=1#c")).name,
    ).toBe("https://example.com/a");
    expect(
      maskCapturedNetworkRequest(request("https://example.com/a#token=1")).name,
    ).toBe("https://example.com/a");
  });

  it("leaves an address with no query as it is", () => {
    expect(
      maskCapturedNetworkRequest(request("https://example.com/api/me")),
    ).toEqual(request("https://example.com/api/me"));
  });

  it("passes on neither the request's body nor the response's", () => {
    const masked = maskCapturedNetworkRequest({
      ...request("https://example.com/api/auth/token"),
      method: "POST",
      requestBody: '{"password":"hunter2"}',
      responseBody: '{"access_token":"abc"}',
    });

    expect(masked.requestBody).toBeUndefined();
    expect(masked.responseBody).toBeUndefined();
    expect(masked.name).toBe("https://example.com/api/auth/token");
    expect(masked.method).toBe("POST");
  });

  it("keeps a relative address without its query", () => {
    expect(
      maskCapturedNetworkRequest(request("/api/proxy/v1/me?token=abc")).name,
    ).toBe("/api/proxy/v1/me");
  });
});
