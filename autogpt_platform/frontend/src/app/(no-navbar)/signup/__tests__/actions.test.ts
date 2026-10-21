import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { TERMS_VERSION } from "@/lib/legal";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { signup } from "../actions";

const mocks = vi.hoisted(() => ({
  captureException: vi.fn(),
  getOnboardingStatus: vi.fn(),
  signUpEmail: vi.fn(),
  rollbackSession: vi.fn(),
  postV1GetOrCreateUser: vi.fn(),
  postV1RecordUserConsent: vi.fn(),
  scheduleAccountCreatedGoal: vi.fn(),
  cookieSet: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: mocks.postV1GetOrCreateUser,
  postV1RecordUserConsent: mocks.postV1RecordUserConsent,
}));
vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: mocks.getOnboardingStatus,
}));
vi.mock("@/lib/auth/auth", () => ({
  auth: { api: { signUpEmail: mocks.signUpEmail } },
}));
vi.mock("@/lib/auth/server/rollbackSession", () => ({
  rollbackSession: mocks.rollbackSession,
}));
vi.mock("next/headers", () => ({
  headers: vi.fn().mockResolvedValue(new Headers()),
  cookies: vi.fn().mockResolvedValue({ set: mocks.cookieSet }),
}));
vi.mock("@/services/analytics/datafast-server", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/analytics/datafast-server")
    >();
  return {
    ...actual,
    scheduleAccountCreatedGoal: mocks.scheduleAccountCreatedGoal,
  };
});
vi.mock("@sentry/nextjs", () => ({
  captureException: mocks.captureException,
}));

describe("email signup account creation tracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Better Auth's signUpEmail sets the session cookie and returns its
    // token when no email verification is required.
    mocks.signUpEmail.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    mocks.getOnboardingStatus.mockResolvedValue({
      shouldShowOnboarding: true,
    });
  });

  it("tracks only a newly created backend account", async () => {
    mocks.postV1GetOrCreateUser.mockResolvedValue({
      status: 200,
      data: {},
      headers: new Headers({ "X-AutoGPT-User-Created": "true" }),
    });

    const result = await signup(
      "new@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      false,
    );

    expect(result.success).toBe(true);
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledWith("email");
    // The browser reports the Google Ads sign-up conversion from this flag on
    // the next page it renders.
    expect(mocks.cookieSet).toHaveBeenCalledWith(
      "agpt_account_created",
      "email",
      expect.objectContaining({ maxAge: 600, path: "/" }),
    );
  });

  it("does not track an account that already existed", async () => {
    mocks.postV1GetOrCreateUser.mockResolvedValue({
      status: 200,
      data: {},
      headers: new Headers({ "X-AutoGPT-User-Created": "false" }),
    });

    const result = await signup(
      "existing@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      false,
    );

    expect(result.success).toBe(true);
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
    expect(mocks.cookieSet).not.toHaveBeenCalled();
    expect(mocks.postV1RecordUserConsent).not.toHaveBeenCalled();
  });

  it("does not track an unverified sign-up; the verification link does", async () => {
    // With AUTH_REQUIRE_EMAIL_VERIFICATION=true there is no session yet, so
    // the account is not provisioned or counted here. /auth/callback?method=email
    // does both once the link is clicked.
    mocks.signUpEmail.mockResolvedValue({ token: null });

    const result = await signup(
      "new@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      true,
    );

    expect(result).toEqual({
      success: true,
      verificationRequired: true,
      email: "new@example.com",
    });
    expect(mocks.postV1GetOrCreateUser).not.toHaveBeenCalled();
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
    expect(mocks.cookieSet).not.toHaveBeenCalled();
  });

  it("reports a thrown backend error instead of completing signup", async () => {
    // The generated client throws ApiError on non-2xx (custom-mutator), which
    // the action catches to roll back the session and surface the failure.
    mocks.postV1GetOrCreateUser.mockRejectedValue({ status: 500 });

    const result = await signup(
      "new@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      false,
    );

    expect(result.success).toBe(false);
    expect(mocks.captureException).toHaveBeenCalledOnce();
    // Revoking the session on provisioning failure is the security-relevant
    // behavior — without it the browser stays authenticated after a failed
    // account setup.
    expect(mocks.rollbackSession).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
    expect(mocks.postV1RecordUserConsent).not.toHaveBeenCalled();
  });
});

describe("email signup consent", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    mocks.signUpEmail.mockResolvedValue({
      token: "session-token",
      user: { id: "user-1" },
    });
    mocks.getOnboardingStatus.mockResolvedValue({
      shouldShowOnboarding: true,
    });
    mocks.postV1RecordUserConsent.mockResolvedValue({ status: 200, data: {} });
  });

  function provisioningResponse(accountCreated: boolean) {
    return {
      status: 200,
      data: {},
      headers: new Headers({
        "X-AutoGPT-User-Created": String(accountCreated),
      }),
    };
  }

  function signupWith(marketingOptOut: boolean) {
    return signup(
      "new@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      marketingOptOut,
    );
  }

  it.each([true, false])(
    "records terms acceptance with marketing_opt_out=%s on a new account",
    async (marketingOptOut) => {
      mocks.postV1GetOrCreateUser.mockResolvedValue(provisioningResponse(true));

      const result = await signupWith(marketingOptOut);

      expect(result).toEqual({ success: true, next: "/onboarding" });
      expect(mocks.postV1RecordUserConsent).toHaveBeenCalledOnce();
      expect(mocks.postV1RecordUserConsent).toHaveBeenCalledWith({
        terms_version: TERMS_VERSION,
        marketing_opt_out: marketingOptOut,
      });
    },
  );

  it("records nothing for an account that already existed", async () => {
    mocks.postV1GetOrCreateUser.mockResolvedValue(provisioningResponse(false));

    const result = await signupWith(true);

    expect(result).toEqual({ success: true, next: "/onboarding" });
    expect(mocks.postV1RecordUserConsent).not.toHaveBeenCalled();
  });

  it.each([
    ["the backend rejects the write", new ApiError("Server error", 500, {})],
    ["the request never arrives", new TypeError("Failed to fetch")],
  ])(
    "still completes signup without a rollback when %s",
    async (_label, error) => {
      mocks.postV1GetOrCreateUser.mockResolvedValue(provisioningResponse(true));
      mocks.postV1RecordUserConsent.mockRejectedValue(error);

      const result = await signupWith(true);

      expect(result).toEqual({ success: true, next: "/onboarding" });
      expect(mocks.rollbackSession).not.toHaveBeenCalled();
      expect(mocks.captureException).toHaveBeenCalledOnce();
      expect(mocks.captureException).toHaveBeenCalledWith(error, {
        tags: { signup_step: "record_consent" },
        user: { id: "user-1" },
        extra: { marketingOptOut: true },
      });
    },
  );
});
