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

import { customMutator } from "../custom-mutator";
import { getSystemHeaders } from "@/lib/impersonation";
import { environment } from "@/services/environment";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";

const mockIsClientSide = vi.mocked(environment.isClientSide);
const mockGetSystemHeaders = vi.mocked(getSystemHeaders);

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
