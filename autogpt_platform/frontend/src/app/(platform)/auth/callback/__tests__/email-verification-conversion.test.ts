import { beforeEach, describe, expect, it, vi } from "vitest";

// The sign_up conversion across the email verification round trip: the
// sign-up action (no session yet) must not count the account, the confirm
// page's click landing on /auth/callback?method=email must count it, and only
// once.

const mocks = vi.hoisted(() => ({
  signUpEmail: vi.fn(),
  verifyEmail: vi.fn(),
  getServerSession: vi.fn(),
  scheduleAccountCreatedGoal: vi.fn(),
  cookieSet: vi.fn(),
  recordUserConsent: vi.fn(),
  provisionedUserIDs: new Set<string>(),
  // The cookie the emailed link left for the confirm page.
  heldLink: "",
}));

const RedirectSignal = vi.hoisted(
  () =>
    class RedirectSignal {
      constructor(readonly url: string) {}
    },
);

vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: { signUpEmail: mocks.signUpEmail, verifyEmail: mocks.verifyEmail },
  },
}));
vi.mock("@/lib/auth/server/getServerSession", () => ({
  getServerSession: mocks.getServerSession,
}));
vi.mock("@/lib/auth/server/rollbackSession", () => ({
  rollbackSession: vi.fn(),
}));

// The backend's POST /auth/user: creates the platform user on the first call
// with a session and says so in X-AutoGPT-User-Created.
vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: vi.fn(async () => {
    const session = await mocks.getServerSession();
    if (!session)
      throw Object.assign(new Error("Unauthorized"), { status: 401 });
    const created = !mocks.provisionedUserIDs.has(session.user.id);
    mocks.provisionedUserIDs.add(session.user.id);
    return {
      status: 200,
      data: {},
      headers: new Headers({ "X-AutoGPT-User-Created": String(created) }),
    };
  }),
  postV1RecordUserConsent: mocks.recordUserConsent,
}));

vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: vi.fn(async () => ({ shouldShowOnboarding: true })),
}));
vi.mock("@/services/analytics/datafast-server", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/analytics/datafast-server")
  >()),
  scheduleAccountCreatedGoal: mocks.scheduleAccountCreatedGoal,
}));
vi.mock("next/headers", () => ({
  headers: vi.fn(async () => new Headers()),
  cookies: vi.fn(async () => ({
    set: mocks.cookieSet,
    get: vi.fn((name: string) =>
      name === "email_verification_link"
        ? { value: mocks.heldLink }
        : undefined,
    ),
    // Clears the held link only as a browser would: same name, same path.
    delete: vi.fn((cookie: { name: string; path?: string }) => {
      if (
        cookie.name === "email_verification_link" &&
        cookie.path === "/auth/verify-email"
      ) {
        mocks.heldLink = "";
      }
    }),
  })),
}));
vi.mock("next/cache", () => ({ revalidatePath: vi.fn() }));
vi.mock("next/navigation", () => ({
  redirect: vi.fn((url: string) => {
    throw new RedirectSignal(url);
  }),
}));
vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
  withServerActionInstrumentation: vi.fn(
    (_name: string, _options: unknown, callback: () => unknown) => callback(),
  ),
}));

import { confirmEmail } from "@/app/(no-navbar)/auth/verify-email/actions";
import { signup } from "@/app/(no-navbar)/signup/actions";
import { GET } from "../route";

const origin = "http://localhost:3000";
const password = "ValidPassword123!"; // pragma: allowlist secret

// The confirm page's button: Better Auth verifies the address and signs the
// user in, the held link is cleared, and the action sends them on to the
// link's callbackURL.
async function clickVerify(callbackURL: string) {
  mocks.heldLink = new URLSearchParams({
    token: "link-token",
    callbackURL,
  }).toString();
  mocks.verifyEmail.mockImplementation(async () => {
    mocks.getServerSession.mockResolvedValue({ user: { id: "u-1" } });
    return { status: true };
  });
  const signal = await confirmEmail().catch((thrown: unknown) => thrown);
  if (!(signal instanceof RedirectSignal)) throw new Error("No redirect");
  expect(mocks.verifyEmail).toHaveBeenCalledWith(
    expect.objectContaining({ query: { token: "link-token" } }),
  );
  expect(mocks.heldLink).toBe("");
  return GET(new Request(`${origin}${signal.url}`));
}

describe("sign_up conversion across email verification", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.provisionedUserIDs.clear();
  });

  it("fires once, when the confirm page's click signs the new user in", async () => {
    mocks.signUpEmail.mockResolvedValue({ token: null, user: { id: "u-1" } });
    mocks.getServerSession.mockResolvedValue(null);

    const signupResult = await signup("new@example.com", password, true);

    expect(signupResult).toMatchObject({ verificationRequired: true });
    expect(mocks.scheduleAccountCreatedGoal).not.toHaveBeenCalled();
    expect(mocks.cookieSet).not.toHaveBeenCalled();

    const callbackURL = mocks.signUpEmail.mock.calls[0][0].body.callbackURL;

    const landing = await clickVerify(callbackURL);

    expect(landing.headers.get("location")).toBe(`${origin}/onboarding`);
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledOnce();
    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledWith("email");
    expect(mocks.cookieSet).toHaveBeenCalledOnce();
    expect(mocks.cookieSet).toHaveBeenCalledWith(
      "agpt_account_created",
      "email",
      expect.anything(),
    );

    // A reload of the landing finds the user provisioned.
    await GET(new Request(`${origin}${callbackURL}`));

    expect(mocks.scheduleAccountCreatedGoal).toHaveBeenCalledOnce();
    expect(mocks.cookieSet).toHaveBeenCalledOnce();
  });

  it.each([true, false])(
    "records the sign-up's marketing opt-out (%s) from the link, opened in any browser",
    async (marketingOptOut) => {
      mocks.signUpEmail.mockResolvedValue({ token: null, user: { id: "u-1" } });
      mocks.getServerSession.mockResolvedValue(null);
      mocks.recordUserConsent.mockResolvedValue({ status: 200, data: {} });

      await signup("new@example.com", password, marketingOptOut);

      // No cookie travels with it: only the link the email carries.
      const callbackURL = mocks.signUpEmail.mock.calls[0][0].body.callbackURL;

      await clickVerify(callbackURL);

      expect(mocks.recordUserConsent).toHaveBeenCalledOnce();
      expect(mocks.recordUserConsent).toHaveBeenCalledWith(
        expect.objectContaining({ marketing_opt_out: marketingOptOut }),
      );
    },
  );
});
