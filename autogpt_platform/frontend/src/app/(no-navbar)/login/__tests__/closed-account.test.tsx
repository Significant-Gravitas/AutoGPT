import type { ReactNode } from "react";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

const mocks = vi.hoisted(() => ({
  login: vi.fn(),
  searchParams: new URLSearchParams(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => mocks.searchParams,
  usePathname: () => "/login",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

vi.mock("../actions", () => ({
  login: mocks.login,
}));

const CLOSED_ACCOUNT_MESSAGE =
  "This account is closed. If you think this is a mistake, contact contact@agpt.co.";
const BANNED_ACCOUNT_MESSAGE =
  "This account has been banned. If you think this is a mistake, please contact support.";

function renderLoginPage(query = "") {
  mocks.searchParams = new URLSearchParams(query);
  render(
    <>
      <LoginPage />
      <Toaster />
    </>,
  );
}

describe("LoginPage for a closed account", () => {
  beforeEach(() => {
    mocks.login.mockReset();
  });

  test("a password sign-in shows the closed-account message, not a generic login error", async () => {
    mocks.login.mockResolvedValue({
      success: false,
      error: CLOSED_ACCOUNT_MESSAGE,
    });
    renderLoginPage();

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "closed@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "hunter2-password" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByText(CLOSED_ACCOUNT_MESSAGE)).toBeDefined();
    expect(screen.queryByText("Login failed")).toBeNull();
    expect(screen.getByRole("button", { name: "Log in" })).toBeDefined();
  });

  test("a social sign-in refused for a closed account lands here with the closed-account message", async () => {
    // Better Auth's OAuth callback sends a banned account to errorCallbackURL
    // with its error code and message in the query.
    renderLoginPage(
      `error=ACCOUNT_CLOSED&error_description=${encodeURIComponent(CLOSED_ACCOUNT_MESSAGE)}`,
    );

    expect(await screen.findByText(CLOSED_ACCOUNT_MESSAGE)).toBeDefined();
  });

  test("shows the app's own copy, never the description from the query", async () => {
    renderLoginPage(
      "error=ACCOUNT_CLOSED&error_description=Call+555-0100+to+restore+access",
    );

    expect(await screen.findByText(CLOSED_ACCOUNT_MESSAGE)).toBeDefined();
    expect(screen.queryByText(/555-0100/)).toBeNull();
  });

  test("a social sign-in refused for another ban says banned, not closed", async () => {
    renderLoginPage(
      "error=BANNED_USER&error_description=You+have+been+banned+for+spam",
    );

    expect(await screen.findByText(BANNED_ACCOUNT_MESSAGE)).toBeDefined();
    expect(screen.queryByText(CLOSED_ACCOUNT_MESSAGE)).toBeNull();
    expect(screen.queryByText(/for spam/)).toBeNull();
  });

  test("a password sign-in refused for another ban shows the banned message", async () => {
    mocks.login.mockResolvedValue({
      success: false,
      error: BANNED_ACCOUNT_MESSAGE,
    });
    renderLoginPage();

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "banned@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "hunter2-password" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByText(BANNED_ACCOUNT_MESSAGE)).toBeDefined();
    expect(screen.queryByText(CLOSED_ACCOUNT_MESSAGE)).toBeNull();
  });

  test("another sign-in error code shows no closed-account message", async () => {
    renderLoginPage("error=signup_disabled&error_description=nope");

    expect(await screen.findByLabelText("Email")).toBeDefined();
    expect(screen.queryByText(CLOSED_ACCOUNT_MESSAGE)).toBeNull();
  });
});
