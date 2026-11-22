import type { ReactNode } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

// SECRT-3384: `/login?next=/home` is where every signed-out visit to the app
// root ends up. For a user who has not finished onboarding, that `next` must
// not win over onboarding.

const mockLoginAction = vi.hoisted(() => vi.fn());
const navigation = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => navigation.searchParams,
  usePathname: () => "/login",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

vi.mock("../actions", () => ({
  login: mockLoginAction,
}));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useToast: () => ({ toasts: [], toast: vi.fn(), dismiss: vi.fn() }),
}));

const EMAIL = "someone@example.com";
const PASSWORD = "hunter2-password";

const originalLocation = window.location;
let pageLocation = { origin: "http://localhost", href: "http://localhost/" };

function submitLoginForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: EMAIL },
  });
  fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
    target: { value: PASSWORD },
  });
  fireEvent.click(screen.getByRole("button", { name: "Log in" }));
}

function openLogin(next: string | null) {
  navigation.searchParams = new URLSearchParams(next ? { next } : {});
  render(<LoginPage />);
}

describe("LoginPage routes an unfinished onboarding ahead of a default landing", () => {
  beforeEach(() => {
    mockLoginAction.mockReset();
    pageLocation = { origin: "http://localhost", href: "http://localhost/" };
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: pageLocation,
    });
  });

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: originalLocation,
    });
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  test.each([
    ["/home", "/home"],
    ["/", "/"],
    ["/copilot", "/home"],
  ])(
    "logging in from ?next=%s before finishing onboarding goes to /onboarding",
    async (next, safeNext) => {
      mockLoginAction.mockResolvedValue({ success: true, next: "/onboarding" });
      openLogin(next);

      submitLoginForm();

      await waitFor(() => expect(pageLocation.href).toBe("/onboarding"));
      expect(mockLoginAction).toHaveBeenCalledWith(EMAIL, PASSWORD, safeNext);
    },
  );

  test("a campaign link keeps its tracking params on the page login lands on", async () => {
    mockLoginAction.mockResolvedValue({ success: true, next: "/onboarding" });
    openLogin("/home?utm_source=newsletter&gclid=abc");

    submitLoginForm();

    await waitFor(() =>
      expect(pageLocation.href).toBe(
        "/onboarding?utm_source=newsletter&gclid=abc",
      ),
    );
  });

  test("an onboarded user logging in from ?next=/home still lands on /home", async () => {
    mockLoginAction.mockResolvedValue({ success: true, next: "/home" });
    openLogin("/home");

    submitLoginForm();

    await waitFor(() => expect(pageLocation.href).toBe("/home"));
  });

  test("a real deep link is still honoured", async () => {
    mockLoginAction.mockResolvedValue({ success: true, next: "/onboarding" });
    openLogin("/library");

    submitLoginForm();

    await waitFor(() => expect(pageLocation.href).toBe("/library"));
    expect(mockLoginAction).toHaveBeenCalledWith(EMAIL, PASSWORD, "/library");
  });

  test("the Sign up link keeps ?next so the round trip is unchanged", () => {
    openLogin("/home");

    expect(
      screen.getByRole("link", { name: /sign up/i }).getAttribute("href"),
    ).toBe("/signup?next=%2Fhome");
  });

  test("continuing with Google from ?next=/home hands /auth/callback the next to decide from", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => new Response(JSON.stringify({})));
    openLogin("/home");

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
    expect(redirectTo).toBe("http://localhost/auth/callback?next=%2Fhome");
  });
});
