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

describe("LoginPage after a banned account is signed out", () => {
  test("says why", () => {
    mockSearchParams = new URLSearchParams({ account: "banned" });

    render(<LoginPage />);

    expect(
      screen.getByText(
        "This account has been banned. If you think this is a mistake, please contact support.",
      ),
    ).toBeDefined();
  });
});
