import type { ReactNode } from "react";
import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, test, vi } from "vitest";
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

afterEach(() => {
  vi.unstubAllEnvs();
  mockSearchParams = new URLSearchParams();
});

describe("Continue with Microsoft on the login page", () => {
  test("shows only where the server has Microsoft sign-in set up", () => {
    const { unmount } = render(<LoginPage />);
    expect(
      screen.queryByRole("button", { name: /continue with microsoft/i }),
    ).toBeNull();
    unmount();

    vi.stubEnv("NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED", "true");
    render(<LoginPage />);
    expect(
      screen.getByRole("button", { name: /continue with microsoft/i }),
    ).toBeDefined();
  });

  test("explains a sign-up Microsoft could not vouch for", () => {
    mockSearchParams = new URLSearchParams({ error: "unconfirmed_email" });

    render(<LoginPage />);

    expect(
      screen.getByText(
        /Microsoft hasn't confirmed the email address on that account/,
      ),
    ).toBeDefined();
  });
});
