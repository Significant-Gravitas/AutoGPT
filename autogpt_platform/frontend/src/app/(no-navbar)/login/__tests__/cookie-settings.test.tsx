import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  configureCookiebot,
  installCookiebot,
  removeCookiebot,
} from "@/tests/integrations/cookiebot";
import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import LoginPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("../actions", () => ({
  login: vi.fn(),
}));

describe("LoginPage cookie settings", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: false,
      isLoggedIn: false,
    });
  });

  afterEach(() => {
    removeCookiebot();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  test("opens the Cookiebot dialog from the Cookie settings link", async () => {
    configureCookiebot();
    const { renew } = installCookiebot({ statistics: true });

    render(<LoginPage />);

    fireEvent.click(
      await screen.findByRole("button", { name: "Cookie settings" }),
    );

    expect(renew).toHaveBeenCalledOnce();
  });

  test("shows the link once Cookiebot asks a visitor with no answer yet", async () => {
    // uc.js loads after hydration, and can arrive after the page finished
    // loading; a visitor with no answer gets no consent event, only the
    // dialog's.
    configureCookiebot();
    vi.spyOn(document, "readyState", "get").mockReturnValue("complete");

    render(<LoginPage />);
    expect(
      await screen.findByText("Log in to your account to continue"),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Cookie settings" }),
    ).toBeNull();

    const { renew } = installCookiebot();
    act(() => {
      window.dispatchEvent(new Event("CookiebotOnDialogInit"));
    });
    fireEvent.click(
      await screen.findByRole("button", { name: "Cookie settings" }),
    );

    expect(renew).toHaveBeenCalledOnce();
  });

  test("hides the link when no consent banner is configured", async () => {
    vi.stubEnv("NEXT_PUBLIC_COOKIEBOT_CBID", "");

    render(<LoginPage />);

    expect(
      await screen.findByText("Log in to your account to continue"),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Cookie settings" }),
    ).toBeNull();
  });

  test("hides the link when Cookiebot could not load", async () => {
    configureCookiebot();
    vi.spyOn(document, "readyState", "get").mockReturnValue("complete");

    render(<LoginPage />);

    expect(
      await screen.findByText("Log in to your account to continue"),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Cookie settings" }),
    ).toBeNull();
  });
});
