import type { ReactNode } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import SignupPage from "../page";

// SECRT-3384: signed out, `/` → `/home` → `/login?next=/home` → "Sign up"
// leaves the user on `/signup?next=/home`. That `next` only names the default
// landing page, so it must not send a brand-new account to Home (with its
// paywall and out-of-credits prompts) ahead of onboarding.

const routerReplace = vi.hoisted(() => vi.fn());
const mockSignupAction = vi.hoisted(() => vi.fn());
const navigation = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: routerReplace,
    push: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => navigation.searchParams,
  usePathname: () => "/signup",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

vi.mock("../actions", () => ({
  signup: mockSignupAction,
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useToast: () => ({ toasts: [], toast: vi.fn(), dismiss: vi.fn() }),
}));

const EMAIL = "new@example.com";
const PASSWORD = "validpassword123";

function submitSignupForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: EMAIL },
  });
  fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
    target: { value: PASSWORD },
  });
  fireEvent.change(
    screen.getByLabelText("Confirm Password", { selector: "input" }),
    { target: { value: PASSWORD } },
  );
  fireEvent.click(screen.getByRole("button", { name: "Sign up" }));
}

function openSignup(next: string | null) {
  navigation.searchParams = new URLSearchParams(next ? { next } : {});
  render(<SignupPage />);
}

describe("SignupPage sends a new account to onboarding first", () => {
  beforeEach(() => {
    routerReplace.mockReset();
    mockSignupAction.mockReset();
    mockSignupAction.mockResolvedValue({ success: true, next: "/onboarding" });
  });

  afterEach(() => {
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  test.each([
    ["/home", "/home"],
    ["/", "/"],
    ["/copilot", "/home"],
    ["/home/", "/home/"],
    ["/copilot/", "/home"],
  ])(
    "signing up from ?next=%s goes straight to /onboarding, never Home",
    async (next, safeNext) => {
      openSignup(next);

      submitSignupForm();

      await waitFor(() =>
        expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
      );
      expect(routerReplace).toHaveBeenCalledTimes(1);
      // The verification email's link goes through /auth/callback, which
      // routes by onboarding state for a default landing too.
      expect(mockSignupAction).toHaveBeenCalledWith(
        EMAIL,
        PASSWORD,
        PASSWORD,
        false,
        safeNext,
      );
    },
  );

  test.each([
    ["/home?utm_source=newsletter", "/onboarding?utm_source=newsletter"],
    [
      "/copilot?utm_campaign=launch&gclid=abc",
      "/onboarding?utm_campaign=launch&gclid=abc",
    ],
  ])(
    "a campaign link (?next=%s) still lands on onboarding with its tracking params",
    async (next, expected) => {
      openSignup(next);

      submitSignupForm();

      await waitFor(() => expect(routerReplace).toHaveBeenCalledWith(expected));
    },
  );

  test("a plain /signup still goes to /onboarding", async () => {
    openSignup(null);

    submitSignupForm();

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
  });

  test.each([
    "/marketplace/agent/autogpt/some-workflow",
    "/library",
    "/home?sessionId=abc",
  ])("a real deep link (?next=%s) is still honoured", async (next) => {
    openSignup(next);

    submitSignupForm();

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith(next));
    expect(mockSignupAction).toHaveBeenCalledWith(
      EMAIL,
      PASSWORD,
      PASSWORD,
      false,
      next,
    );
  });

  test("continuing with Google from a campaign link hands /auth/callback the whole next", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response(JSON.stringify({})));
    openSignup("/home?utm_source=newsletter");

    fireEvent.click(
      screen.getByRole("button", { name: /continue with google/i }),
    );

    await waitFor(() =>
      expect(fetchSpy).toHaveBeenCalledWith(
        "/api/auth/login/with-provider",
        expect.anything(),
      ),
    );
    const [, init] = fetchSpy.mock.calls.find(
      ([input]) => input === "/api/auth/login/with-provider",
    )!;
    const { redirectTo } = JSON.parse(String(init?.body));
    expect(redirectTo).toBe(
      `${window.location.origin}/auth/callback?next=${encodeURIComponent("/home?utm_source=newsletter")}`,
    );
  });
});
