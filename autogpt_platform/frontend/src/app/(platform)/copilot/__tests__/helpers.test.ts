import { beforeEach, describe, expect, it, vi } from "vitest";
import { IMPERSONATION_HEADER_NAME } from "@/lib/constants";
import { getCopilotAuthHeaders } from "../helpers";

vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: vi.fn(),
}));

vi.mock("@/lib/impersonation", () => ({
  ImpersonationState: {
    get: vi.fn(),
  },
}));

import { getWebSocketToken } from "@/lib/supabase/actions";
import { ImpersonationState } from "@/lib/impersonation";

const mockGetWebSocketToken = vi.mocked(getWebSocketToken);
const mockImpersonationStateGet = vi.mocked(ImpersonationState.get);

describe("getCopilotAuthHeaders", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockImpersonationStateGet.mockReturnValue(null);
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
    mockImpersonationStateGet.mockReturnValue("impersonated-user-123");

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
