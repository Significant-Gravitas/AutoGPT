import { beforeEach, describe, expect, it, vi } from "vitest";
import { GET } from "./route";

const mocks = vi.hoisted(() => ({
  getOnboardingStatus: vi.fn(),
  getServerSession: vi.fn(),
  postV1GetOrCreateUser: vi.fn(),
  revalidatePath: vi.fn(),
  scheduleAccountCreatedGoal: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: mocks.postV1GetOrCreateUser,
}));
vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: mocks.getOnboardingStatus,
}));
vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: mocks.getServerSession,
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
vi.mock("next/cache", () => ({ revalidatePath: mocks.revalidatePath }));

describe("OAuth callback account creation tracking", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubEnv("NODE_ENV", "development");
    // Better Auth sets the session cookie before redirecting here; the route
    // only verifies the session, so a truthy user drives the create path.
    mocks.getServerSession.mockResolvedValue({ user: { id: "u1" } });
    mocks.getOnboardingStatus.mockResolvedValue({
      shouldShowOnboarding: false,
    });
  });

  it("tracks a newly created Google account", async () => {
    mocks.postV1GetOrCreateUser.mockResolvedValue({
      status: 200,
      data: {},
      headers: new Headers({ "X-AutoGPT-User-Created": "true" }),
    });

    const response = await GET(
      new Request("http://localhost/auth/callback?code=valid"),
    );

    expect(response.headers.get("location")).toBe("http://localhost/copilot");
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledWith("google");
  });

  it("does not track a returning Google user", async () => {
    mocks.postV1GetOrCreateUser.mockResolvedValue({
      status: 200,
      data: {},
      headers: new Headers({ "X-AutoGPT-User-Created": "false" }),
    });

    await GET(new Request("http://localhost/auth/callback?code=valid"));

    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
  });

  it("redirects a thrown backend error without tracking", async () => {
    // The generated client throws ApiError on non-2xx (custom-mutator).
    mocks.postV1GetOrCreateUser.mockRejectedValue({ status: 500 });

    const response = await GET(
      new Request("http://localhost/auth/callback?code=valid"),
    );

    expect(response.headers.get("location")).toBe(
      "http://localhost/error?message=server-error",
    );
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
  });
});
