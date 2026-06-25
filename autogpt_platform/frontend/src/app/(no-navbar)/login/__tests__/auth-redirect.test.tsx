import { render, waitFor } from "@testing-library/react";
import { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

const routerReplace = vi.fn();
const routerPush = vi.fn();
let mockSearchParams = new URLSearchParams();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: routerReplace,
    push: routerPush,
    refresh: vi.fn(),
  }),
  useSearchParams: () => mockSearchParams,
  usePathname: () => "/login",
}));

let mockIsLoggedIn = true;
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({
    supabase: {},
    user: { id: "u" },
    isUserLoading: false,
    isLoggedIn: mockIsLoggedIn,
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("../actions", () => ({
  login: vi.fn(),
}));

describe("LoginPage auth-redirect useEffect", () => {
  beforeEach(() => {
    routerReplace.mockClear();
    routerPush.mockClear();
    mockSearchParams = new URLSearchParams();
    mockIsLoggedIn = true;
  });

  test("with a safe ?next= deep link, redirects there via router.replace", async () => {
    mockSearchParams = new URLSearchParams({ next: "/library" });
    render(<LoginPage />);
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/library"));
    expect(routerPush).not.toHaveBeenCalled();
  });

  test("with no ?next=, defers to OnboardingProvider instead of bouncing through /", async () => {
    render(<LoginPage />);
    await new Promise((r) => setTimeout(r, 20));
    expect(routerReplace).not.toHaveBeenCalled();
    expect(routerPush).not.toHaveBeenCalled();
  });

  test("an unsafe ?next= (absolute URL) is dropped — no redirect", async () => {
    mockSearchParams = new URLSearchParams({ next: "//evil.site" });
    render(<LoginPage />);
    await new Promise((r) => setTimeout(r, 20));
    expect(routerReplace).not.toHaveBeenCalled();
  });
});
