import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();
const postV1GetOrCreateUserMock = vi.fn();
const getOnboardingStatusMock = vi.fn();
const revalidatePathMock = vi.fn();
const scheduleAccountCreatedGoalMock = vi.fn();

vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: (...args: unknown[]) =>
    postV1GetOrCreateUserMock(...args),
}));

// Keep the real wasAccountCreated (it reads the provisioning response header);
// only the goal dispatch, which hits cookies and the network, is stubbed.
vi.mock("@/services/analytics/datafast-server", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/analytics/datafast-server")
  >()),
  scheduleAccountCreatedGoal: (...args: unknown[]) =>
    scheduleAccountCreatedGoalMock(...args),
}));

vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: () => getOnboardingStatusMock(),
}));

vi.mock("next/cache", () => ({
  revalidatePath: (...args: unknown[]) => revalidatePathMock(...args),
}));

import { GET } from "../route";

class BackendApiErrorStub extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "BackendApiErrorStub";
    this.status = status;
  }
}

const origin = "http://localhost:3000";

function makeCallbackRequest(
  path = "/auth/callback",
  headers: Record<string, string> = {},
): Request {
  return new Request(`${origin}${path}`, { headers });
}

// Shape of the backend provisioning call the route feeds to wasAccountCreated.
function provisioningResponse(accountCreated = false) {
  return {
    status: 200,
    data: {},
    headers: new Headers({
      "X-AutoGPT-User-Created": String(accountCreated),
    }),
  };
}

function loggedInWithCompletedSetup() {
  getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
  postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse());
  getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });
}

beforeEach(() => {
  getServerSessionMock.mockReset();
  postV1GetOrCreateUserMock.mockReset();
  getOnboardingStatusMock.mockReset();
  revalidatePathMock.mockReset();
  scheduleAccountCreatedGoalMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("auth callback GET — session handling", () => {
  it("redirects to the auth-code-error page when there is no session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const response = await GET(makeCallbackRequest());

    expect(response.status).toBe(307);
    expect(response.headers.get("location")).toBe(
      `${origin}/auth/auth-code-error`,
    );
    expect(postV1GetOrCreateUserMock).not.toHaveBeenCalled();
  });

  it("sends fresh users to onboarding and revalidates the layout", async () => {
    vi.stubEnv("NODE_ENV", "development");
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse());
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(`${origin}/onboarding`);
    expect(revalidatePathMock).toHaveBeenCalledWith("/onboarding", "layout");
  });

  it("sends already-onboarded users to copilot", async () => {
    vi.stubEnv("NODE_ENV", "development");
    loggedInWithCompletedSetup();

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(`${origin}/copilot`);
    expect(revalidatePathMock).toHaveBeenCalledWith("/copilot", "layout");
  });
});

describe("auth callback GET — account creation tracking", () => {
  beforeEach(() => {
    vi.stubEnv("NODE_ENV", "development");
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });
  });

  it("tracks a newly created Google account", async () => {
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(`${origin}/copilot`);
    expect(scheduleAccountCreatedGoalMock).toHaveBeenCalledOnce();
    expect(scheduleAccountCreatedGoalMock).toHaveBeenCalledWith("google");
  });

  it("does not track a returning Google user", async () => {
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(false));

    await GET(makeCallbackRequest());

    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
  });
});

describe("auth callback GET — redirect target resolution", () => {
  it("honors the next query parameter in development without consulting x-forwarded-host", async () => {
    vi.stubEnv("NODE_ENV", "development");
    loggedInWithCompletedSetup();

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace", {
        "x-forwarded-host": "app.example.com",
      }),
    );

    expect(response.headers.get("location")).toBe(`${origin}/marketplace`);
  });

  it("redirects through the forwarded host in production", async () => {
    vi.stubEnv("NODE_ENV", "production");
    loggedInWithCompletedSetup();

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace", {
        "x-forwarded-host": "app.example.com",
      }),
    );

    expect(response.headers.get("location")).toBe(
      "https://app.example.com/marketplace",
    );
  });

  it("falls back to the request origin in production when no forwarded host is set", async () => {
    vi.stubEnv("NODE_ENV", "production");
    loggedInWithCompletedSetup();

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace"),
    );

    expect(response.headers.get("location")).toBe(`${origin}/marketplace`);
  });

  it("ignores an off-site next and cannot open-redirect via the forwarded host", async () => {
    vi.stubEnv("NODE_ENV", "production");
    loggedInWithCompletedSetup(); // shouldShowOnboarding: false -> /copilot

    for (const evil of ["@evil.com", "//evil.com", "https://evil.com"]) {
      const response = await GET(
        makeCallbackRequest(`/auth/callback?next=${encodeURIComponent(evil)}`, {
          "x-forwarded-host": "app.example.com",
        }),
      );
      // sanitizeAuthNext drops the crafted value, so we land on the resolved
      // in-app target under our own host — never evil.com.
      expect(response.headers.get("location")).toBe(
        "https://app.example.com/copilot",
      );
    }
  });
});

describe("auth callback GET — user creation failures", () => {
  beforeEach(() => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
  });

  it("redirects to auth-token-invalid when the backend rejects with 401", async () => {
    postV1GetOrCreateUserMock.mockRejectedValue(
      new BackendApiErrorStub("Unauthorized", 401),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=auth-token-invalid`,
    );
  });

  it("redirects to server-error when the backend rejects with a 5xx status", async () => {
    postV1GetOrCreateUserMock.mockRejectedValue(
      new BackendApiErrorStub("Internal Server Error", 500),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=server-error`,
    );
    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
  });

  it("redirects to rate-limited when the backend rejects with 429", async () => {
    postV1GetOrCreateUserMock.mockRejectedValue(
      new BackendApiErrorStub("Too Many Requests", 429),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=rate-limited`,
    );
  });

  it("redirects to network-error when the fetch itself fails", async () => {
    postV1GetOrCreateUserMock.mockRejectedValue(
      new TypeError("Failed to fetch"),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=network-error`,
    );
  });

  it("redirects to user-creation-failed for any other failure", async () => {
    postV1GetOrCreateUserMock.mockRejectedValue(
      new Error("something else broke"),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=user-creation-failed`,
    );
  });
});
