import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: vi.fn(() => true),
    isClientSide: vi.fn(() => false),
    getAGPTServerApiUrl: vi.fn(() => "http://localhost:8006/api"),
  },
}));

import {
  buildUrlWithQuery,
  createRequestHeaders,
  makeAuthenticatedFileUpload,
} from "../helpers";
import { environment } from "@/services/environment";
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
      [API_KEY_HEADER_NAME]: "api-key-value", // pragma: allowlist secret
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

describe("makeAuthenticatedFileUpload", () => {
  beforeEach(() => {
    vi.mocked(environment.isClientSide).mockReturnValue(true);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.mocked(environment.isClientSide).mockReturnValue(false);
  });

  it.each([
    [413, "Appearance images must be 5 MB or smaller."],
    [400, "Choose a PNG, JPEG or WebP image."],
    [
      422,
      "This image wasn't approved for an Expert appearance. Choose another image.",
    ],
    [503, "Image storage is unavailable. Please try again later."],
  ])(
    "preserves actionable backend details for HTTP %i",
    async (status, detail) => {
      vi.stubGlobal(
        "fetch",
        vi.fn().mockResolvedValue(Response.json({ detail }, { status })),
      );

      await expect(
        makeAuthenticatedFileUpload("/upload", new FormData()),
      ).rejects.toMatchObject({
        name: "ApiError",
        message: detail,
        status,
        response: { detail },
      });
    },
  );

  it("keeps the general upload limit for a bodyless 413", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(null, { status: 413, statusText: "Payload Too Large" }),
        ),
    );

    await expect(
      makeAuthenticatedFileUpload("/upload", new FormData()),
    ).rejects.toMatchObject({
      message: "File is too large — max size is 256MB",
      status: 413,
    });
  });

  it("provides recovery guidance when the error has no body or status text", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(new Response(null, { status: 502 })),
    );

    await expect(
      makeAuthenticatedFileUpload("/upload", new FormData()),
    ).rejects.toMatchObject({
      message: "Request failed (HTTP 502). Please try again.",
      status: 502,
      response: null,
    });
  });

  it("falls back to HTTP status text for a non-JSON error", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response("unavailable", {
          status: 502,
          statusText: "Bad Gateway",
        }),
      ),
    );

    await expect(
      makeAuthenticatedFileUpload("/upload", new FormData()),
    ).rejects.toMatchObject({
      message: "Bad Gateway",
      status: 502,
      response: null,
    });
  });
});
