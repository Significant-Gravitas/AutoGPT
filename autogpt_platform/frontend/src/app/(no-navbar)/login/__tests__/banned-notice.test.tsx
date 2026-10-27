import type { ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

let mockSearchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => mockSearchParams,
  usePathname: () => "/login",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

describe("LoginPage after a refused account is signed out", () => {
  test("says why", () => {
    mockSearchParams = new URLSearchParams({ account: "banned" });

    render(<LoginPage />);

    expect(
      screen.getByText(
        "This account has been banned. If you think this is a mistake, please contact support.",
      ),
    ).toBeDefined();
  });

  test("says the account is closed when it was signed out after deleting", () => {
    mockSearchParams = new URLSearchParams({ account: "closed" });

    render(<LoginPage />);

    expect(
      screen.getByText(
        "This account is closed. If you think this is a mistake, contact contact@agpt.co.",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/has been banned/)).toBeNull();
  });
});
