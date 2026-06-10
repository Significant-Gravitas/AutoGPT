import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();
const createUserMock = vi.fn();
const getOnboardingStatusMock = vi.fn();
const revalidatePathMock = vi.fn();

vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
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

function loggedInWithCompletedSetup() {
  getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
  createUserMock.mockResolvedValue({ id: "user-1" });
  getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });
}

beforeEach(() => {
  getServerSessionMock.mockReset();
  createUserMock.mockReset();
  getOnboardingStatusMock.mockReset();
  revalidatePathMock.mockReset();
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
    expect(createUserMock).not.toHaveBeenCalled();
  });

  it("sends fresh users to onboarding and revalidates the layout", async () => {
    vi.stubEnv("NODE_ENV", "development");
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    createUserMock.mockResolvedValue({ id: "user-1" });
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
});

describe("auth callback GET — user creation failures", () => {
  beforeEach(() => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
  });

  it("redirects to auth-token-invalid when the backend rejects with 401", async () => {
    createUserMock.mockRejectedValue(
      new BackendApiErrorStub("Unauthorized", 401),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=auth-token-invalid`,
    );
  });

  it("redirects to server-error when the backend rejects with a 5xx status", async () => {
    createUserMock.mockRejectedValue(
      new BackendApiErrorStub("Internal Server Error", 500),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=server-error`,
    );
  });

  it("redirects to rate-limited when the backend rejects with 429", async () => {
    createUserMock.mockRejectedValue(
      new BackendApiErrorStub("Too Many Requests", 429),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=rate-limited`,
    );
  });

  it("redirects to network-error when the fetch itself fails", async () => {
    createUserMock.mockRejectedValue(new TypeError("Failed to fetch"));

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=network-error`,
    );
  });

  it("redirects to user-creation-failed for any other failure", async () => {
    createUserMock.mockRejectedValue(new Error("something else broke"));

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=user-creation-failed`,
    );
  });
});
