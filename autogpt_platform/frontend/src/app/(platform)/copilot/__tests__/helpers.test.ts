import { beforeEach, describe, expect, it, vi } from "vitest";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import { getCopilotAuthHeaders } from "../helpers";

vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: vi.fn(),
}));

vi.mock("@/lib/impersonation", () => ({
  getSystemHeaders: vi.fn(),
}));

import { getWebSocketToken } from "@/lib/supabase/actions";
import { getSystemHeaders } from "@/lib/impersonation";

const mockGetWebSocketToken = vi.mocked(getWebSocketToken);
const mockGetSystemHeaders = vi.mocked(getSystemHeaders);

describe("getCopilotAuthHeaders", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockGetSystemHeaders.mockReturnValue({});
  });

  it("returns Authorization header when token is present and no impersonation active", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: "test-jwt-token",
      error: undefined,
    });

    const headers = await getCopilotAuthHeaders();

    expect(headers).toEqual({ Authorization: "Bearer test-jwt-token" });
  });

  it("includes X-Act-As-User-Id header when impersonation is active", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: "test-jwt-token",
      error: undefined,
    });
    mockGetSystemHeaders.mockReturnValue({
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-123",
    });

    const headers = await getCopilotAuthHeaders();

    expect(headers).toEqual({
      Authorization: "Bearer test-jwt-token",
      [IMPERSONATION_HEADER_NAME]: "impersonated-user-123",
    });
  });

  it("throws when getWebSocketToken returns an error", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: null,
      error: "Token fetch failed",
    });

    await expect(getCopilotAuthHeaders()).rejects.toThrow(
      "Authentication failed — please sign in again.",
    );
  });

  it("throws when getWebSocketToken returns no token and no error", async () => {
    mockGetWebSocketToken.mockResolvedValue({
      token: null,
      error: undefined,
    });

    await expect(getCopilotAuthHeaders()).rejects.toThrow(
      "Authentication failed — please sign in again.",
    );
  });
});
