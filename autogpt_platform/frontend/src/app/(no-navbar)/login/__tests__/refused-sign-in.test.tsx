import type { ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

const mocks = vi.hoisted(() => ({
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

const BANNED_NOTICE =
  "This account has been banned. If you think this is a mistake, please contact support.";

function renderLoginPage(query: string) {
  mocks.searchParams = new URLSearchParams(query);
  render(<LoginPage />);
}

describe("LoginPage after a refused social sign-in", () => {
  test("a banned account lands here with the banned notice", async () => {
    // Better Auth's OAuth callback sends a banned account to errorCallbackURL
    // with its error code and message in the query.
    renderLoginPage("error=BANNED_USER&error_description=You+have+been+banned");

    expect(await screen.findByText(BANNED_NOTICE)).toBeDefined();
  });

  test("shows the app's own copy, never the description from the query", async () => {
    renderLoginPage(
      "error=BANNED_USER&error_description=Call+555-0100+to+restore+access",
    );

    expect(await screen.findByText(BANNED_NOTICE)).toBeDefined();
    expect(screen.queryByText(/555-0100/)).toBeNull();
  });

  test("another sign-in error code shows no notice", async () => {
    renderLoginPage("error=signup_disabled&error_description=nope");

    expect(await screen.findByLabelText("Email")).toBeDefined();
    expect(screen.queryByText(BANNED_NOTICE)).toBeNull();
  });
});
