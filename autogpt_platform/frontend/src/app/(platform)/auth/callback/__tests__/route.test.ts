import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { TERMS_VERSION } from "@/lib/legal";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();
const postV1GetOrCreateUserMock = vi.fn();
const postV1RecordUserConsentMock = vi.fn();
const rollbackSessionMock = vi.fn();
const getOnboardingStatusMock = vi.fn();
const revalidatePathMock = vi.fn();
const scheduleAccountCreatedGoalMock = vi.fn();
const cookieSetMock = vi.fn();
const cookieGetMock = vi.fn();
const cookieDeleteMock = vi.fn();
const captureExceptionMock = vi.fn();

vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: (...args: unknown[]) =>
    postV1GetOrCreateUserMock(...args),
  postV1RecordUserConsent: (...args: unknown[]) =>
    postV1RecordUserConsentMock(...args),
}));

vi.mock("@/lib/auth/server/rollbackSession", () => ({
  rollbackSession: () => rollbackSessionMock(),
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

vi.mock("next/headers", () => ({
  cookies: () =>
    Promise.resolve({
      set: cookieSetMock,
      get: cookieGetMock,
      delete: cookieDeleteMock,
    }),
  headers: () => new Headers(),
}));

vi.mock("next/cache", () => ({
  revalidatePath: (...args: unknown[]) => revalidatePathMock(...args),
}));

vi.mock("@sentry/nextjs", () => ({
  captureException: (...args: unknown[]) => captureExceptionMock(...args),
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
  postV1RecordUserConsentMock.mockReset().mockResolvedValue({ status: 200 });
  rollbackSessionMock.mockReset();
  cookieSetMock.mockReset();
  cookieGetMock.mockReset();
  cookieDeleteMock.mockReset();
  captureExceptionMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("auth callback GET — session handling", () => {
  it("redirects to the auth-code-error page when there is no session", async () => {
    vi.stubEnv("BETTER_AUTH_URL", "https://autogpt.example.com");
    getServerSessionMock.mockResolvedValue(null);

    const response = await GET(makeCallbackRequest());

    expect(response.status).toBe(307);
    expect(response.headers.get("location")).toBe(
      "https://autogpt.example.com/auth/auth-code-error",
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
    // The browser reports the Google Ads sign-up conversion from this flag on
    // the page it lands on.
    expect(cookieSetMock).toHaveBeenCalledWith(
      "agpt_account_created",
      "google",
      expect.objectContaining({ maxAge: 600, path: "/" }),
    );
  });

  it("does not track a returning Google user", async () => {
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(false));

    await GET(makeCallbackRequest());

    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
    expect(cookieSetMock).not.toHaveBeenCalled();
  });
});

describe("auth callback GET — signup consent", () => {
  const optOutCookie = { name: "agpt_marketing_opt_out", value: "1" };

  beforeEach(() => {
    vi.stubEnv("NODE_ENV", "development");
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });
  });

  function withOptOutCookie() {
    cookieGetMock.mockImplementation((name: string) =>
      name === optOutCookie.name ? optOutCookie : undefined,
    );
  }

  it("records a marketing refusal carried by the cookie and clears it", async () => {
    withOptOutCookie();
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(`${origin}/onboarding`);
    expect(postV1RecordUserConsentMock).toHaveBeenCalledOnce();
    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: true,
    });
    expect(cookieDeleteMock).toHaveBeenCalledWith("agpt_marketing_opt_out");
  });

  it("records terms acceptance without a refusal when there is no cookie", async () => {
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));

    await GET(makeCallbackRequest());

    expect(postV1RecordUserConsentMock).toHaveBeenCalledOnce();
    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: false,
    });
    expect(cookieDeleteMock).not.toHaveBeenCalled();
  });

  it("records the refusal on a returning account and clears the cookie", async () => {
    withOptOutCookie();
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(false));

    await GET(makeCallbackRequest());

    expect(postV1RecordUserConsentMock).toHaveBeenCalledOnce();
    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: true,
    });
    expect(cookieDeleteMock).toHaveBeenCalledWith("agpt_marketing_opt_out");
    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
  });

  it("records nothing for a returning account that did not opt out", async () => {
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(false));

    await GET(makeCallbackRequest());

    expect(postV1RecordUserConsentMock).not.toHaveBeenCalled();
    expect(cookieDeleteMock).not.toHaveBeenCalled();
  });

  it("keeps the refusal for the retry when provisioning fails", async () => {
    withOptOutCookie();
    postV1GetOrCreateUserMock.mockRejectedValue(new Error("backend down"));

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      `${origin}/error?message=user-creation-failed`,
    );
    expect(rollbackSessionMock).toHaveBeenCalledOnce();
    expect(cookieDeleteMock).not.toHaveBeenCalled();
    expect(postV1RecordUserConsentMock).not.toHaveBeenCalled();
  });

  it("still lands the user on the normal next page when the consent write fails", async () => {
    withOptOutCookie();
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));
    const error = new ApiError("Internal Server Error", 500, {});
    postV1RecordUserConsentMock.mockRejectedValue(error);

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace"),
    );

    expect(response.headers.get("location")).toBe(`${origin}/marketplace`);
    expect(revalidatePathMock).toHaveBeenCalledWith("/marketplace", "layout");
    expect(rollbackSessionMock).not.toHaveBeenCalled();
    expect(captureExceptionMock).toHaveBeenCalledWith(error, {
      tags: { signup_step: "record_consent" },
      user: { id: "user-1" },
      extra: { marketingOptOut: true },
    });
  });
});

describe("auth callback GET — email verification link", () => {
  beforeEach(() => {
    vi.stubEnv("NODE_ENV", "development");
  });

  it("provisions and tracks a new email account once the link signs it in", async () => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    const response = await GET(
      makeCallbackRequest("/auth/callback?method=email"),
    );

    expect(response.headers.get("location")).toBe(`${origin}/onboarding`);
    expect(postV1GetOrCreateUserMock).toHaveBeenCalledOnce();
    expect(scheduleAccountCreatedGoalMock).toHaveBeenCalledOnce();
    expect(scheduleAccountCreatedGoalMock).toHaveBeenCalledWith("email");
    expect(cookieSetMock).toHaveBeenCalledOnce();
    expect(cookieSetMock).toHaveBeenCalledWith(
      "agpt_account_created",
      "email",
      expect.objectContaining({ maxAge: 600, path: "/" }),
    );
  });

  it("records the terms and a refusal carried from /signup once the link signs a new account in", async () => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    cookieGetMock.mockImplementation((name: string) =>
      name === "agpt_marketing_opt_out"
        ? { name: "agpt_marketing_opt_out", value: "1" }
        : undefined,
    );
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    await GET(makeCallbackRequest("/auth/callback?method=email"));

    expect(postV1RecordUserConsentMock).toHaveBeenCalledOnce();
    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: true,
    });
    expect(cookieDeleteMock).toHaveBeenCalledWith("agpt_marketing_opt_out");
  });

  it("records a refusal carried by the link when it creates the account", async () => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(true));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    await GET(
      makeCallbackRequest("/auth/callback?method=email&marketing_opt_out=1"),
    );

    expect(postV1RecordUserConsentMock).toHaveBeenCalledOnce();
    expect(postV1RecordUserConsentMock).toHaveBeenCalledWith({
      terms_version: TERMS_VERSION,
      marketing_opt_out: true,
    });
  });

  it.each([
    ["an existing account", "/auth/callback?method=email&marketing_opt_out=1"],
    ["a Google sign-in", "/auth/callback?marketing_opt_out=1"],
  ])("ignores the link's refusal for %s", async (_, path) => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue(
      provisioningResponse(path.includes("method=email") ? false : true),
    );
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: true });

    await GET(makeCallbackRequest(path));

    const calls = postV1RecordUserConsentMock.mock.calls;
    expect(calls.every(([body]) => body.marketing_opt_out === false)).toBe(
      true,
    );
  });

  it("does not track an existing account that verifies at its next sign-in", async () => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
    postV1GetOrCreateUserMock.mockResolvedValue(provisioningResponse(false));
    getOnboardingStatusMock.mockResolvedValue({ shouldShowOnboarding: false });

    const response = await GET(
      makeCallbackRequest("/auth/callback?method=email&next=/marketplace"),
    );

    expect(response.headers.get("location")).toBe(`${origin}/marketplace`);
    expect(scheduleAccountCreatedGoalMock).not.toHaveBeenCalled();
    expect(cookieSetMock).not.toHaveBeenCalled();
  });

  it("sends an expired or used link to log in, keeping next", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const response = await GET(
      makeCallbackRequest(
        "/auth/callback?method=email&next=%2Fmarketplace&error=TOKEN_EXPIRED",
      ),
    );

    expect(response.headers.get("location")).toBe(
      `${origin}/login?email_verification=expired&next=%2Fmarketplace`,
    );
    expect(postV1GetOrCreateUserMock).not.toHaveBeenCalled();
  });

  it("sends a click on an already verified link to log in", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const response = await GET(
      makeCallbackRequest("/auth/callback?method=email&next=https://evil.com"),
    );

    expect(response.headers.get("location")).toBe(
      `${origin}/login?email_verification=verified`,
    );
  });
});

describe("auth callback GET — redirect target resolution", () => {
  it("uses the canonical HTTPS origin and ignores a malicious forwarded host", async () => {
    vi.stubEnv("NODE_ENV", "development");
    vi.stubEnv("BETTER_AUTH_URL", "https://autogpt.example.com");
    loggedInWithCompletedSetup();

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace", {
        "x-forwarded-host": "evil.example.com",
      }),
    );

    expect(response.headers.get("location")).toBe(
      "https://autogpt.example.com/marketplace",
    );
  });

  it("preserves a canonical HTTP origin for LAN deployments", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("BETTER_AUTH_URL", "http://192.168.1.20:3000");
    loggedInWithCompletedSetup();

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace", {
        "x-forwarded-host": "evil.example.com",
      }),
    );

    expect(response.headers.get("location")).toBe(
      "http://192.168.1.20:3000/marketplace",
    );
  });

  it("falls back to NEXT_PUBLIC_FRONTEND_BASE_URL for the canonical origin", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("NEXT_PUBLIC_FRONTEND_BASE_URL", "https://public.example.com");
    loggedInWithCompletedSetup();

    const response = await GET(
      makeCallbackRequest("/auth/callback?next=/marketplace", {
        "x-forwarded-host": "evil.example.com",
      }),
    );

    expect(response.headers.get("location")).toBe(
      "https://public.example.com/marketplace",
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

  it("ignores an off-site next and cannot open-redirect via forwarded headers", async () => {
    vi.stubEnv("NODE_ENV", "production");
    vi.stubEnv("BETTER_AUTH_URL", "https://autogpt.example.com");
    loggedInWithCompletedSetup(); // shouldShowOnboarding: false -> /copilot

    for (const evil of ["@evil.com", "//evil.com", "https://evil.com"]) {
      const response = await GET(
        makeCallbackRequest(`/auth/callback?next=${encodeURIComponent(evil)}`, {
          "x-forwarded-host": "evil.example.com",
        }),
      );
      // sanitizeAuthNext drops the crafted value, so we land on the resolved
      // in-app target under our own host — never evil.com.
      expect(response.headers.get("location")).toBe(
        "https://autogpt.example.com/copilot",
      );
    }
  });
});

describe("auth callback GET — user creation failures", () => {
  beforeEach(() => {
    getServerSessionMock.mockResolvedValue({ user: { id: "user-1" } });
  });

  it("redirects to auth-token-invalid when the backend rejects with 401", async () => {
    vi.stubEnv("BETTER_AUTH_URL", "https://autogpt.example.com");
    postV1GetOrCreateUserMock.mockRejectedValue(
      new BackendApiErrorStub("Unauthorized", 401),
    );

    const response = await GET(makeCallbackRequest());

    expect(response.headers.get("location")).toBe(
      "https://autogpt.example.com/error?message=auth-token-invalid",
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
    expect(rollbackSessionMock).toHaveBeenCalledOnce();
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
