import { beforeEach, describe, expect, it, vi } from "vitest";
import { signup } from "../actions";

const mocks = vi.hoisted(() => ({
  captureException: vi.fn(),
  getOnboardingStatus: vi.fn(),
  getServerSupabase: vi.fn(),
  postV1GetOrCreateUser: vi.fn(),
  scheduleAccountCreatedGoal: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: mocks.postV1GetOrCreateUser,
}));
vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: mocks.getOnboardingStatus,
}));
vi.mock("@/lib/supabase/server/getServerSupabase", () => ({
  getServerSupabase: mocks.getServerSupabase,
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
    mocks.getServerSupabase.mockResolvedValue({
      auth: {
        signUp: vi.fn().mockResolvedValue({
          data: { session: { access_token: "token" } },
          error: null,
        }),
        setSession: vi.fn().mockResolvedValue(undefined),
      },
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

  it("reports a resolved backend error instead of completing signup", async () => {
    mocks.postV1GetOrCreateUser.mockResolvedValue({
      status: 500,
      data: {},
      headers: new Headers(),
    });

    const result = await signup(
      "new@example.com",
      "ValidPassword123!",
      "ValidPassword123!",
      true,
    );

    expect(result.success).toBe(false);
    expect(mocks.captureException).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
  });
});
