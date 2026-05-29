import { describe, expect, it, vi } from "vitest";

vi.mock("@/lib/supabase/server/getServerSupabase", () => ({
  getServerSupabase: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => true),
    isClientSide: vi.fn(() => false),
    getAGPTServerApiUrl: vi.fn(() => "http://localhost:8006/api"),
  },
}));

import { buildUrlWithQuery, createRequestHeaders } from "../helpers";
import {
  API_KEY_HEADER_NAME,
  IMPERSONATION_HEADER_NAME,
} from "@/lib/constants";

function makeRequest(headers: Record<string, string>): Request {
  return new Request("http://example.com/test", { headers });
}

describe("buildUrlWithQuery", () => {
  const url = "http://example.com/api";

  it("returns the URL unchanged when query is undefined", () => {
    expect(buildUrlWithQuery(url)).toBe(url);
  });

  it("returns the URL unchanged when every value is null or undefined", () => {
    expect(buildUrlWithQuery(url, { a: null, b: undefined })).toBe(url);
  });

  it("filters out null and undefined values but keeps falsy primitives", () => {
    const result = buildUrlWithQuery(url, {
      kept: "yes",
      zero: 0,
      empty: "",
      flag: false,
      missing: undefined,
      blank: null,
    });

    const params = new URL(result).searchParams;
    expect(params.get("kept")).toBe("yes");
    expect(params.get("zero")).toBe("0");
    expect(params.get("empty")).toBe("");
    expect(params.get("flag")).toBe("false");
    expect(params.has("missing")).toBe(false);
    expect(params.has("blank")).toBe(false);
  });
});

describe("createRequestHeaders — basics", () => {
  it("adds Content-Type when hasRequestBody is true", () => {
    const headers = createRequestHeaders("token-abc", true);
    expect(headers["Content-Type"]).toBe("application/json");
  });

  it("omits Content-Type when hasRequestBody is false", () => {
    const headers = createRequestHeaders("token-abc", false);
    expect(headers["Content-Type"]).toBeUndefined();
  });

  it("uses the provided contentType override", () => {
    const headers = createRequestHeaders(
      "token-abc",
      true,
      "application/x-www-form-urlencoded",
    );
    expect(headers["Content-Type"]).toBe("application/x-www-form-urlencoded");
  });

  it("adds Authorization header when token is a real value", () => {
    const headers = createRequestHeaders("token-abc", false);
    expect(headers["Authorization"]).toBe("Bearer token-abc");
  });

  it("omits Authorization when token is null", () => {
    const headers = createRequestHeaders(null, false);
    expect(headers["Authorization"]).toBeUndefined();
  });

  it("omits Authorization when token is empty", () => {
    const headers = createRequestHeaders("", false);
    expect(headers["Authorization"]).toBeUndefined();
  });
});

describe("createRequestHeaders — Sentry trace forwarding", () => {
  it("forwards sentry-trace and baggage headers when present on originalRequest", () => {
    const request = makeRequest({
      "sentry-trace": "0123456789abcdef0123456789abcdef-0123456789abcdef-1",
      baggage: "sentry-environment=local,sentry-public_key=abc",
    });

    const headers = createRequestHeaders(
      "token-abc",
      false,
      undefined,
      request,
    );

    expect(headers["sentry-trace"]).toBe(
      "0123456789abcdef0123456789abcdef-0123456789abcdef-1",
    );
    expect(headers["baggage"]).toBe(
      "sentry-environment=local,sentry-public_key=abc",
    );
  });

  it("forwards only sentry-trace when baggage is absent", () => {
    const request = makeRequest({
      "sentry-trace": "trace-id-only",
    });

    const headers = createRequestHeaders(
      "token-abc",
      false,
      undefined,
      request,
    );

    expect(headers["sentry-trace"]).toBe("trace-id-only");
    expect(headers["baggage"]).toBeUndefined();
  });

  it("forwards only baggage when sentry-trace is absent", () => {
    const request = makeRequest({
      baggage: "sentry-environment=prod",
    });

    const headers = createRequestHeaders(
      "token-abc",
      false,
      undefined,
      request,
    );

    expect(headers["sentry-trace"]).toBeUndefined();
    expect(headers["baggage"]).toBe("sentry-environment=prod");
  });

  it("does not forward sentry headers when originalRequest has none", () => {
    const request = makeRequest({ "X-Other-Header": "something" });

    const headers = createRequestHeaders(
      "token-abc",
      false,
      undefined,
      request,
    );

    expect(headers["sentry-trace"]).toBeUndefined();
    expect(headers["baggage"]).toBeUndefined();
  });

  it("does not attempt to forward sentry headers when originalRequest is omitted", () => {
    const headers = createRequestHeaders("token-abc", false);

    expect(headers["sentry-trace"]).toBeUndefined();
    expect(headers["baggage"]).toBeUndefined();
  });
});

describe("createRequestHeaders — impersonation and API-key forwarding", () => {
  it("forwards the impersonation header alongside sentry headers", () => {
    const request = makeRequest({
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-xyz",
      "sentry-trace": "trace-id",
    });

    const headers = createRequestHeaders(
      "token-abc",
      false,
      undefined,
      request,
    );

    expect(headers[IMPERSONATION_HEADER_NAME]).toBe("impersonated-user-xyz");
    expect(headers["sentry-trace"]).toBe("trace-id");
  });

  it("forwards the API key header alongside sentry headers", () => {
    const request = makeRequest({
      [API_KEY_HEADER_NAME]: "api-key-value",
      baggage: "sentry-environment=local",
    });

    const headers = createRequestHeaders(
      "token-abc",
      false,
      undefined,
      request,
    );

    expect(headers[API_KEY_HEADER_NAME]).toBe("api-key-value");
    expect(headers["baggage"]).toBe("sentry-environment=local");
  });
});
