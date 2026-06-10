import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const signUpEmailMock = vi.fn();
const rollbackSessionMock = vi.fn();
const postV1GetOrCreateUserMock = vi.fn();
const getOnboardingStatusMock = vi.fn();
const isWaitlistErrorMock = vi.fn();
const logWaitlistErrorMock = vi.fn();
const captureExceptionMock = vi.fn();

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      signUpEmail: (...args: unknown[]) => signUpEmailMock(...args),
    },
  },
}));

vi.mock("@/lib/auth/server/rollbackSession", () => ({
  rollbackSession: (...args: unknown[]) => rollbackSessionMock(...args),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: (...args: unknown[]) =>
    postV1GetOrCreateUserMock(...args),
}));

vi.mock("@/app/api/helpers", async (importActual) => {
  const actual = await importActual<typeof import("@/app/api/helpers")>();
  return {
    ...actual,
    getOnboardingStatus: () => getOnboardingStatusMock(),
  };
});

vi.mock("@/app/api/auth/utils", () => ({
  isWaitlistError: (...args: unknown[]) => isWaitlistErrorMock(...args),
  logWaitlistError: (...args: unknown[]) => logWaitlistErrorMock(...args),
}));

vi.mock("next/headers", () => ({
  headers: vi.fn(async () => new Headers()),
}));

vi.mock("@sentry/nextjs", () => ({
  captureException: (...args: unknown[]) => captureExceptionMock(...args),
}));

import { signup } from "../actions";

const email = "new.user@example.com";
const validPassword = "a-long-enough-password";

function signupWithValidPayload() {
  return signup(email, validPassword, validPassword, true);
}

beforeEach(() => {
  signUpEmailMock.mockReset();
  rollbackSessionMock.mockReset();
  postV1GetOrCreateUserMock.mockReset();
  getOnboardingStatusMock.mockReset();
  isWaitlistErrorMock.mockReset().mockReturnValue(false);
  logWaitlistErrorMock.mockReset();
  captureExceptionMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("signup", () => {
  it("creates the account, provisions the backend user, and routes to onboarding", async () => {
    signUpEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue({
      status: 200,
      data: { id: "user-1" },
    });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await signupWithValidPayload();

    expect(signUpEmailMock).toHaveBeenCalledWith({
      body: { email, password: validPassword, name: "new.user" },
      headers: expect.any(Headers),
    });
    expect(postV1GetOrCreateUserMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("routes straight to copilot when onboarding is already complete", async () => {
    signUpEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue({
      status: 200,
      data: { id: "user-1" },
    });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const result = await signupWithValidPayload();

    expect(result).toEqual({ success: true, next: "/copilot" });
  });

  it("reports user_already_exists when Better Auth rejects a duplicate email", async () => {
    signUpEmailMock.mockRejectedValue(
      new APIError("UNPROCESSABLE_ENTITY", {
        message: "User already exists",
        code: "USER_ALREADY_EXISTS",
      }),
    );

    const result = await signupWithValidPayload();

    expect(result).toEqual({ success: false, error: "user_already_exists" });
    expect(postV1GetOrCreateUserMock).not.toHaveBeenCalled();
  });

  it("reports not_allowed when the failure is a waitlist rejection", async () => {
    isWaitlistErrorMock.mockReturnValue(true);
    signUpEmailMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: 'The email address "[email]" is not allowed to register.',
        code: "P0001",
      }),
    );

    const result = await signupWithValidPayload();

    expect(result).toEqual({ success: false, error: "not_allowed" });
    expect(logWaitlistErrorMock).toHaveBeenCalledWith(
      "Signup",
      expect.any(String),
    );
  });

  it("surfaces the Better Auth message for other APIErrors", async () => {
    signUpEmailMock.mockRejectedValue(
      new APIError("BAD_REQUEST", {
        message: "Password is too weak",
        code: "WEAK_PASSWORD",
      }),
    );

    const result = await signupWithValidPayload();

    expect(result).toEqual({ success: false, error: "Password is too weak" });
  });

  it("asks the user to retry when backend user provisioning fails after sign-up", async () => {
    signUpEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockRejectedValue(new Error("backend down"));

    const result = await signupWithValidPayload();

    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
    expect(rollbackSessionMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      success: false,
      error: "Failed to complete account setup. Please try again.",
    });
  });

  it("rejects a password shorter than 12 characters without calling Better Auth", async () => {
    const result = await signup(email, "short-pass", "short-pass", true);

    expect(result).toEqual({ success: false, error: "Invalid signup payload" });
    expect(signUpEmailMock).not.toHaveBeenCalled();
  });
});
