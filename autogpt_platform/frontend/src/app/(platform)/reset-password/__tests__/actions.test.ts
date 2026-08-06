import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const requestPasswordResetMock = vi.fn();
const resetPasswordMock = vi.fn();
const redirectMock = vi.fn();

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      requestPasswordReset: (...args: unknown[]) =>
        requestPasswordResetMock(...args),
      resetPassword: (...args: unknown[]) => resetPasswordMock(...args),
    },
  },
}));

vi.mock("@sentry/nextjs", () => ({
  withServerActionInstrumentation: (
    _name: string,
    _options: object,
    fn: () => unknown,
  ) => fn(),
}));

vi.mock("next/navigation", () => ({
  redirect: (...args: unknown[]) => redirectMock(...args),
}));

import { changePassword, sendResetEmail } from "../actions";

beforeEach(() => {
  requestPasswordResetMock.mockReset();
  resetPasswordMock.mockReset();
  redirectMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("sendResetEmail", () => {
  it("requests a reset link that redirects back to the configured origin", async () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://platform.agpt.test");
    requestPasswordResetMock.mockResolvedValue({ status: true });

    const result = await sendResetEmail("user@example.com");

    expect(requestPasswordResetMock).toHaveBeenCalledWith({
      body: {
        email: "user@example.com",
        redirectTo: "https://platform.agpt.test/reset-password",
      },
    });
    expect(result).toBeUndefined();
  });

  it("falls back to localhost when no frontend base URL is configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "");
    requestPasswordResetMock.mockResolvedValue({ status: true });

    await sendResetEmail("user@example.com");

    expect(requestPasswordResetMock).toHaveBeenCalledWith({
      body: {
        email: "user@example.com",
        redirectTo: "http://localhost:3000/reset-password",
      },
    });
  });

  it("returns the Better Auth message when the reset request fails with an APIError", async () => {
    requestPasswordResetMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: "Email rate limit exceeded",
        code: "RATE_LIMITED",
      }),
    );

    const result = await sendResetEmail("user@example.com");

    expect(result).toBe("Email rate limit exceeded");
  });

  it("returns a generic message when the reset request fails unexpectedly", async () => {
    requestPasswordResetMock.mockRejectedValue(new Error("smtp down"));

    const result = await sendResetEmail("user@example.com");

    expect(result).toBe("Failed to send reset email. Please try again.");
  });
});

describe("changePassword", () => {
  it("resets the password with the token and redirects to login", async () => {
    resetPasswordMock.mockResolvedValue({ status: true });

    const result = await changePassword("new-secure-password", "token-123");

    expect(resetPasswordMock).toHaveBeenCalledWith({
      body: { newPassword: "new-secure-password", token: "token-123" },
    });
    expect(redirectMock).toHaveBeenCalledWith("/login");
    expect(result).toBeUndefined();
  });

  it("returns the Better Auth message without redirecting when the token is rejected", async () => {
    resetPasswordMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: "Invalid or expired token",
        code: "INVALID_TOKEN",
      }),
    );

    const result = await changePassword("new-secure-password", "bad-token");

    expect(result).toBe("Invalid or expired token");
    expect(redirectMock).not.toHaveBeenCalled();
  });

  it("returns a generic message without redirecting on unexpected failures", async () => {
    resetPasswordMock.mockRejectedValue(new Error("db unavailable"));

    const result = await changePassword("new-secure-password", "token-123");

    expect(result).toBe("Failed to change password. Please try again.");
    expect(redirectMock).not.toHaveBeenCalled();
  });
});
