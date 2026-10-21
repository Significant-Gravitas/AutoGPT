import { TERMS_VERSION } from "@/lib/legal";
import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const signInEmailMock = vi.fn();
const rollbackSessionMock = vi.fn();
const createUserMock = vi.fn();
const getOnboardingStatusMock = vi.fn();
const captureExceptionMock = vi.fn();
const scheduleAccountCreatedGoalMock = vi.fn();
const markAccountCreatedMock = vi.fn();
const recordUserConsentMock = vi.fn();

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      signInEmail: (...args: unknown[]) => signInEmailMock(...args),
    },
  },
}));

vi.mock("@/lib/auth/server/rollbackSession", () => ({
  rollbackSession: (...args: unknown[]) => rollbackSessionMock(...args),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: (...args: unknown[]) => createUserMock(...args),
  postV1RecordUserConsent: (...args: unknown[]) =>
    recordUserConsentMock(...args),
}));

vi.mock("@/services/analytics/datafast-server", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/analytics/datafast-server")
  >()),
  scheduleAccountCreatedGoal: (...args: unknown[]) =>
    scheduleAccountCreatedGoalMock(...args),
}));

vi.mock("@/services/analytics/account-created-server", () => ({
  markAccountCreated: (...args: unknown[]) => markAccountCreatedMock(...args),
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

// What the backend's get-or-create answers: X-AutoGPT-User-Created is true
// only on the call that created the User.
function userResponse(created: boolean, status = 200) {
  return {
    status,
    data: { id: "user-1" },
    headers: new Headers({ "X-AutoGPT-User-Created": String(created) }),
  };
}

beforeEach(() => {
  signInEmailMock.mockReset();
  rollbackSessionMock.mockReset();
  createUserMock.mockReset();
  getOnboardingStatusMock.mockReset();
  captureExceptionMock.mockReset();
  scheduleAccountCreatedGoalMock.mockReset();
  markAccountCreatedMock.mockReset();
  recordUserConsentMock.mockReset().mockResolvedValue({ status: 200 });
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("login", () => {
  it("signs in, provisions the backend user, and points new users at onboarding", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(false));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await login("user@example.com", "hunter2-password");

    expect(signInEmailMock).toHaveBeenCalledWith({
      body: {
        email: "user@example.com",
        password: "hunter2-password",
        callbackURL: "/auth/callback?method=email",
      },
      headers: expect.any(Headers),
    });
    expect(createUserMock).toHaveBeenCalledTimes(1);
    expect(getOnboardingStatusMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("sends returning users to /home when onboarding is already complete", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(false));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const result = await login("user@example.com", "hunter2-password");

    expect(result).toEqual({ success: true, next: "/home" });
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

  it("reports email_not_verified so the page can show check-your-inbox", async () => {
    // Right password, unverified address: with sendOnSignIn Better Auth has
    // already emailed a fresh link by the time it throws this.
    signInEmailMock.mockRejectedValue(
      new APIError("FORBIDDEN", {
        message: "Email not verified",
        code: "EMAIL_NOT_VERIFIED",
      }),
    );

    const result = await login(
      "unverified@example.com",
      "hunter2-password",
      "/marketplace",
    );

    expect(result).toEqual({
      success: false,
      error: "email_not_verified",
      email: "unverified@example.com",
    });
    expect(signInEmailMock.mock.calls[0][0].body.callbackURL).toBe(
      "/auth/callback?method=email&next=%2Fmarketplace",
    );
    expect(createUserMock).not.toHaveBeenCalled();
    expect(rollbackSessionMock).not.toHaveBeenCalled();
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
    expect(rollbackSessionMock).toHaveBeenCalledTimes(1);
    expect(result).toEqual({
      success: false,
      error: "Failed to login. Please try again.",
    });
  });

  it("does not roll back the session when login succeeds end to end", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(false));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const result = await login("user@example.com", "hunter2-password");

    expect(rollbackSessionMock).not.toHaveBeenCalled();
    expect(result).toEqual({ success: true, next: "/home" });
  });

  it("counts the sign-up when this login is the call that created the account", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(true));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await login("user@example.com", "hunter2-password");

    expect(scheduleAccountCreatedGoalMock).toHaveBeenCalledTimes(1);
    expect(scheduleAccountCreatedGoalMock).toHaveBeenCalledWith("email");
    expect(markAccountCreatedMock).toHaveBeenCalledTimes(1);
    expect(markAccountCreatedMock).toHaveBeenCalledWith("email");
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("records the terms when this login is the call that created the account", async () => {
    // A mail scanner opened the verification link first, so the owner's own
    // sign-in is what sets the account up.
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(true));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    await login("user@example.com", "hunter2-password");

    expect(recordUserConsentMock).toHaveBeenCalledTimes(1);
    expect(recordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: false,
    });
  });

  it("still signs the new account in when the consent write fails", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(true));
    recordUserConsentMock.mockRejectedValue(new Error("backend down"));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const result = await login("user@example.com", "hunter2-password");

    expect(rollbackSessionMock).not.toHaveBeenCalled();
    expect(result).toEqual({ success: true, next: "/onboarding" });
  });

  it("does not count a login into an account that already existed", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(false));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    await login("user@example.com", "hunter2-password");

    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
    expect(markAccountCreatedMock).not.toHaveBeenCalled();
    expect(recordUserConsentMock).not.toHaveBeenCalled();
  });

  it("rolls back and counts nothing when provisioning answers an error status", async () => {
    signInEmailMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue(userResponse(true, 500));

    const result = await login("user@example.com", "hunter2-password");

    expect(rollbackSessionMock).toHaveBeenCalledTimes(1);
    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
    expect(markAccountCreatedMock).not.toHaveBeenCalled();
    expect(result).toEqual({
      success: false,
      error: "Failed to login. Please try again.",
    });
  });
});
