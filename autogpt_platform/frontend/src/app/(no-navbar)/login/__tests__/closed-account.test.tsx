import type { ReactNode } from "react";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

const mockLoginAction = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
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

const CLOSED_ACCOUNT_MESSAGE =
  "This account is closed. If you think this is a mistake, contact contact@agpt.co.";

describe("LoginPage for a closed account", () => {
  beforeEach(() => {
    mockLoginAction.mockReset();
  });

  test("shows the closed-account message, not a generic login error", async () => {
    mockLoginAction.mockResolvedValue({
      success: false,
      error: CLOSED_ACCOUNT_MESSAGE,
    });
    render(
      <>
        <LoginPage />
        <Toaster />
      </>,
    );

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
});
