import { beforeEach, describe, expect, it, vi } from "vitest";
import { signup } from "../actions";

const mocks = vi.hoisted(() => ({
  captureException: vi.fn(),
  getOnboardingStatus: vi.fn(),
  signUpEmail: vi.fn(),
  rollbackSession: vi.fn(),
  postV1GetOrCreateUser: vi.fn(),
  scheduleAccountCreatedGoal: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: mocks.postV1GetOrCreateUser,
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
    // Better Auth's signUpEmail sets the session cookie; a resolved call is
    // the success case.
    mocks.signUpEmail.mockResolvedValue({});
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
      true,
    );

    expect(result.success).toBe(true);
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledWith("email");
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
      true,
    );

    expect(result.success).toBe(true);
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
  });

  it("reports a thrown backend error instead of completing signup", async () => {
    // The generated client throws ApiError on non-2xx (custom-mutator), which
    // the action catches to roll back the session and surface the failure.
    mocks.postV1GetOrCreateUser.mockRejectedValue({ status: 500 });

    const result = await signup(
      "new@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      true,
    );

    expect(result.success).toBe(false);
    expect(mocks.captureException).toHaveBeenCalledOnce();
    // Revoking the session on provisioning failure is the security-relevant
    // behavior — without it the browser stays authenticated after a failed
    // account setup.
    expect(mocks.rollbackSession).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
  });
});
