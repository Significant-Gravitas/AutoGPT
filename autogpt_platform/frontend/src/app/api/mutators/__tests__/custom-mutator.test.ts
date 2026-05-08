import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/impersonation", () => ({
  getSystemHeaders: vi.fn(),
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isClientSide: vi.fn(),
    isServerSide: vi.fn(),
    getAGPTServerBaseUrl: vi.fn(() => "http://localhost:8006"),
  },
}));

vi.mock("@/lib/autogpt-server-api/helpers", () => ({
  ApiError: class ApiError extends Error {
    constructor(
      message: string,
      public status: number,
      public data: unknown,
    ) {
      super(message);
    }
  },
  createRequestHeaders: vi.fn(() => ({})),
  getServerAuthToken: vi.fn(),
}));

vi.mock("@sentry/nextjs", () => ({
  getTraceData: vi.fn(() => ({})),
}));

import { customMutator } from "../custom-mutator";
import { getSystemHeaders } from "@/lib/impersonation";
import { environment } from "@/services/environment";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import * as Sentry from "@sentry/nextjs";

const mockIsClientSide = vi.mocked(environment.isClientSide);
const mockGetSystemHeaders = vi.mocked(getSystemHeaders);
const mockGetTraceData = vi.mocked(Sentry.getTraceData);

describe("customMutator — impersonation header", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsClientSide.mockReturnValue(true);
    mockGetSystemHeaders.mockReturnValue({});
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ "content-type": "application/json" }),
        json: () => Promise.resolve({}),
      }),
    );
  });

  it("adds impersonation header when impersonation is active", async () => {
    mockGetSystemHeaders.mockReturnValue({
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-abc",
    });

    await customMutator("/test", { method: "GET" });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers[IMPERSONATION_HEADER_NAME]).toBe("impersonated-user-abc");
  });

  it("does not add impersonation header when no impersonation active", async () => {
    mockGetSystemHeaders.mockReturnValue({});

    await customMutator("/test", { method: "GET" });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers[IMPERSONATION_HEADER_NAME]).toBeUndefined();
  });

  it("coexists with pre-existing caller-supplied headers without overwriting them", async () => {
    mockGetSystemHeaders.mockReturnValue({
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-abc",
    });

    await customMutator("/test", {
      method: "GET",
      headers: { "X-Custom-Header": "custom-value" },
    });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers[IMPERSONATION_HEADER_NAME]).toBe("impersonated-user-abc");
    expect(headers["X-Custom-Header"]).toBe("custom-value");
  });
});

describe("customMutator — Sentry trace propagation", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsClientSide.mockReturnValue(true);
    mockGetSystemHeaders.mockReturnValue({});
    mockGetTraceData.mockReturnValue({});
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        headers: new Headers({ "content-type": "application/json" }),
        json: () => Promise.resolve({}),
      }),
    );
  });

  it("attaches sentry-trace and baggage headers from Sentry trace data on client-side", async () => {
    mockGetTraceData.mockReturnValue({
      "sentry-trace": "0123456789abcdef0123456789abcdef-0123456789abcdef-1",
      baggage: "sentry-environment=local,sentry-public_key=abc",
    });

    await customMutator("/test", { method: "GET" });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers["sentry-trace"]).toBe(
      "0123456789abcdef0123456789abcdef-0123456789abcdef-1",
    );
    expect(headers["baggage"]).toBe(
      "sentry-environment=local,sentry-public_key=abc",
    );
  });

  it("omits sentry-trace headers when Sentry has no active trace", async () => {
    mockGetTraceData.mockReturnValue({});

    await customMutator("/test", { method: "GET" });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers["sentry-trace"]).toBeUndefined();
    expect(headers["baggage"]).toBeUndefined();
  });

  it("does not attach Sentry trace headers on server-side", async () => {
    mockIsClientSide.mockReturnValue(false);
    mockGetTraceData.mockReturnValue({
      "sentry-trace": "should-not-appear",
    });

    await customMutator("/test", { method: "GET" });

    expect(mockGetTraceData).not.toHaveBeenCalled();
  });

  it("skips non-string values returned by Sentry.getTraceData", async () => {
    // Simulate a non-string slipping into the trace-data object
    mockGetTraceData.mockReturnValue({
      "sentry-trace": "real-trace",
      "sentry-sampled": 1,
    } as unknown as ReturnType<typeof Sentry.getTraceData>);

    await customMutator("/test", { method: "GET" });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers["sentry-trace"]).toBe("real-trace");
    expect(headers["sentry-sampled"]).toBeUndefined();
  });

  it("falls back to an empty object when Sentry.getTraceData is undefined", async () => {
    // Simulate an older @sentry/nextjs build where getTraceData isn't exported
    (Sentry as { getTraceData?: unknown }).getTraceData =
      undefined as unknown as typeof Sentry.getTraceData;

    await customMutator("/test", { method: "GET" });

    const fetchCall = vi.mocked(fetch).mock.calls[0];
    const headers = fetchCall[1]?.headers as Record<string, string>;
    expect(headers["sentry-trace"]).toBeUndefined();
    expect(headers["baggage"]).toBeUndefined();

    // Restore for subsequent tests
    (Sentry as { getTraceData?: unknown }).getTraceData = mockGetTraceData;
  });
});
