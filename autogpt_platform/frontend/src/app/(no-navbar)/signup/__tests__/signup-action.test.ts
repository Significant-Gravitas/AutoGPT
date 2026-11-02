import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { TERMS_VERSION } from "@/lib/legal";
import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const signUpEmailMock = vi.fn();
const rollbackSessionMock = vi.fn();
const postV1GetOrCreateUserMock = vi.fn();
const postV1RecordUserConsentMock = vi.fn();
const wasAccountCreatedMock = vi.fn();
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
  postV1RecordUserConsent: (...args: unknown[]) =>
    postV1RecordUserConsentMock(...args),
}));

// DataFast account tracking is exercised by actions.test.ts; stub it here so
// wasAccountCreated doesn't read .headers off the minimal mocked response.
vi.mock("@/services/analytics/datafast-server", () => ({
  wasAccountCreated: (...args: unknown[]) => wasAccountCreatedMock(...args),
  scheduleAccountCreatedGoal: vi.fn(),
}));

vi.mock("@/services/analytics/account-created-server", () => ({
  markAccountCreated: vi.fn(),
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
  return signup(email, validPassword, validPassword, false);
}

beforeEach(() => {
  signUpEmailMock.mockReset();
  rollbackSessionMock.mockReset();
  postV1GetOrCreateUserMock.mockReset();
  postV1RecordUserConsentMock.mockReset().mockResolvedValue({ status: 200 });
  wasAccountCreatedMock.mockReset().mockReturnValue(false);
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
    signUpEmailMock.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    postV1GetOrCreateUserMock.mockResolvedValue({
      status: 200,
      data: { id: "user-1" },
    });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await signupWithValidPayload();

    expect(signUpEmailMock).toHaveBeenCalledWith({
      body: {
        email,
        password: validPassword,
        name: "new.user",
        callbackURL: "/auth/callback?method=email",
      },
      headers: expect.any(Headers),
    });
    expect(postV1GetOrCreateUserMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("puts a marketing refusal in the verification link", async () => {
    signUpEmailMock.mockResolvedValue({ token: null, user: { id: "user-1" } });

    await signup(email, validPassword, validPassword, true, "/marketplace");

    expect(signUpEmailMock.mock.calls[0][0].body.callbackURL).toBe(
      "/auth/callback?method=email&next=%2Fmarketplace&marketing_opt_out=1",
    );
  });

  it("records terms acceptance on the account it just created", async () => {
    signUpEmailMock.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    postV1GetOrCreateUserMock.mockResolvedValue({ status: 200, data: {} });
    wasAccountCreatedMock.mockReturnValue(true);
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await signup(email, validPassword, validPassword, true);

    expect(postV1RecordUserConsentMock).toHaveBeenCalledTimes(1);
    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: true,
    });
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("does not record consent when signing up into an existing account", async () => {
    signUpEmailMock.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    postV1GetOrCreateUserMock.mockResolvedValue({ status: 200, data: {} });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const result = await signup(email, validPassword, validPassword, true);

    expect(postV1RecordUserConsentMock).not.toHaveBeenCalled();
    expect(result).toEqual({ success: true, next: "/home" });
  });

  it("keeps the new account signed in when the consent write fails", async () => {
    signUpEmailMock.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    postV1GetOrCreateUserMock.mockResolvedValue({ status: 200, data: {} });
    wasAccountCreatedMock.mockReturnValue(true);
    postV1RecordUserConsentMock.mockRejectedValue(
      new ApiError("Unprocessable Entity", 422, {}),
    );
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await signupWithValidPayload();

    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: false,
    });
    expect(rollbackSessionMock).not.toHaveBeenCalled();
    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("routes straight to /home when onboarding is already complete", async () => {
    signUpEmailMock.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    postV1GetOrCreateUserMock.mockResolvedValue({
      status: 200,
      data: { id: "user-1" },
    });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const result = await signupWithValidPayload();

    expect(result).toEqual({ success: true, next: "/home" });
  });

  it("reports user_already_exists when Better Auth rejects a duplicate email", async () => {
    // Better Auth's email sign-up throws USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL.
    signUpEmailMock.mockRejectedValue(
      new APIError("UNPROCESSABLE_ENTITY", {
        message: "User already exists. Use another email.",
        code: "USER_ALREADY_EXISTS_USE_ANOTHER_EMAIL",
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
    signUpEmailMock.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    postV1GetOrCreateUserMock.mockRejectedValue(new Error("backend down"));

    const result = await signupWithValidPayload();

    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
    expect(rollbackSessionMock).toHaveBeenCalledTimes(1);
    expect(postV1RecordUserConsentMock).not.toHaveBeenCalled();
    expect(result).toEqual({
      success: false,
      error: "Failed to complete account setup. Please try again.",
    });
  });

  it("shows check-your-inbox instead of provisioning when verification is required", async () => {
    // AUTH_REQUIRE_EMAIL_VERIFICATION=true: Better Auth creates no session and
    // emails a link. Provisioning without a session is what used to fail with
    // 401 "Failed to complete account setup".
    signUpEmailMock.mockResolvedValue({ token: null, user: { id: "user-1" } });

    const result = await signupWithValidPayload();

    expect(result).toEqual({
      success: true,
      verificationRequired: true,
      email,
    });
    expect(postV1GetOrCreateUserMock).not.toHaveBeenCalled();
    expect(rollbackSessionMock).not.toHaveBeenCalled();
    expect(getOnboardingStatusMock).not.toHaveBeenCalled();
  });

  it("carries a safe next path through the verification link", async () => {
    signUpEmailMock.mockResolvedValue({ token: null, user: { id: "user-1" } });

    await signup(email, validPassword, validPassword, false, "/marketplace");
    await signup(
      email,
      validPassword,
      validPassword,
      false,
      "https://evil.com",
    );

    expect(signUpEmailMock.mock.calls[0][0].body.callbackURL).toBe(
      "/auth/callback?method=email&next=%2Fmarketplace",
    );
    expect(signUpEmailMock.mock.calls[1][0].body.callbackURL).toBe(
      "/auth/callback?method=email",
    );
  });

  it("rejects a password shorter than 12 characters without calling Better Auth", async () => {
    const result = await signup(email, "short-pass", "short-pass", false);

    expect(result).toEqual({ success: false, error: "Invalid signup payload" });
    expect(signUpEmailMock).not.toHaveBeenCalled();
  });
});
