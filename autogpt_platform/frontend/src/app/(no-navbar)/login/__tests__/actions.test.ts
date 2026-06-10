import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const signInEmailMock = vi.fn();
const createUserMock = vi.fn();
const getOnboardingStatusMock = vi.fn();
const captureExceptionMock = vi.fn();

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      signInEmail: (...args: unknown[]) => signInEmailMock(...args),
    },
  },
}));

vi.mock("@/lib/autogpt-server-api", () => ({
  default: class BackendAPIMock {
    createUser(...args: unknown[]) {
      return createUserMock(...args);
    }
  },
}));

vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: () => getOnboardingStatusMock(),
}));

vi.mock("next/headers", () => ({
  headers: vi.fn(async () => new Headers()),
}));

vi.mock("@sentry/nextjs", () => ({
  captureException: (...args: unknown[]) => captureExceptionMock(...args),
}));

import { login } from "../actions";

beforeEach(() => {
  signInEmailMock.mockReset();
  createUserMock.mockReset();
  getOnboardingStatusMock.mockReset();
  captureExceptionMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("login", () => {
  it("signs in, provisions the backend user, and points new users at onboarding", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue({ id: "user-1" });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await login("user@example.com", "hunter2-password");

    expect(signInEmailMock).toHaveBeenCalledWith({
      body: { email: "user@example.com", password: "hunter2-password" },
      headers: expect.any(Headers),
    });
    expect(createUserMock).toHaveBeenCalledTimes(1);
    expect(getOnboardingStatusMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("sends returning users to copilot when onboarding is already complete", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue({ id: "user-1" });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const result = await login("user@example.com", "hunter2-password");

    expect(result).toEqual({ success: true, next: "/copilot" });
  });

  it("returns the Better Auth error message when sign-in fails with an APIError", async () => {
    signInEmailMock.mockRejectedValue(
      new APIError("UNAUTHORIZED", {
        message: "Invalid credentials",
        code: "INVALID_EMAIL_OR_PASSWORD",
      }),
    );

    const result = await login("user@example.com", "wrong-password");

    expect(result).toEqual({ success: false, error: "Invalid credentials" });
    expect(createUserMock).not.toHaveBeenCalled();
  });

  it("falls back to a generic message when the APIError carries no body message", async () => {
    signInEmailMock.mockRejectedValue(new APIError("UNAUTHORIZED"));

    const result = await login("user@example.com", "wrong-password");

    expect(result).toEqual({
      success: false,
      error: "Invalid email or password",
    });
  });

  it("rejects a malformed email without calling Better Auth", async () => {
    const result = await login("not-an-email", "hunter2-password");

    expect(result).toEqual({
      success: false,
      error: "Invalid email or password",
    });
    expect(signInEmailMock).not.toHaveBeenCalled();
  });

  it("captures unexpected failures in Sentry and returns a generic login error", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockRejectedValue(new Error("backend unreachable"));

    const result = await login("user@example.com", "hunter2-password");

    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      success: false,
      error: "Failed to login. Please try again.",
    });
  });
});
