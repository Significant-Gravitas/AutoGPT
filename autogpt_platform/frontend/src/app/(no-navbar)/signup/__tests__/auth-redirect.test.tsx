import { render, waitFor } from "@testing-library/react";
import { ReactNode } from "react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import SignupPage from "../page";

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
  usePathname: () => "/signup",
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

// Keep OnboardingProvider out of this test — we're isolating the auth-redirect
// useEffect here. Provider behaviour is covered in
// `src/providers/onboarding/__tests__/onboarding-provider-routing.test.tsx`.
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("../actions", () => ({
  signup: vi.fn(),
}));

describe("SignupPage auth-redirect useEffect", () => {
  beforeEach(() => {
    routerReplace.mockClear();
    routerPush.mockClear();
    mockSearchParams = new URLSearchParams();
    mockIsLoggedIn = true;
  });

  test("with a safe ?next= deep link, redirects there via router.replace", async () => {
    mockSearchParams = new URLSearchParams({ next: "/library" });
    render(<SignupPage />);
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/library"));
    expect(routerPush).not.toHaveBeenCalled();
  });

  test("with no ?next= the page stays put and defers to OnboardingProvider", async () => {
    // Before this PR, this case pushed to "/" which bounced through /copilot
    // before OnboardingProvider could redirect — the flash users reported.
    render(<SignupPage />);
    await new Promise((r) => setTimeout(r, 20));
    expect(routerReplace).not.toHaveBeenCalled();
    expect(routerPush).not.toHaveBeenCalled();
  });

  test("an unsafe ?next= (absolute URL) is dropped — no redirect", async () => {
    mockSearchParams = new URLSearchParams({ next: "https://phishing.site" });
    render(<SignupPage />);
    await new Promise((r) => setTimeout(r, 20));
    expect(routerReplace).not.toHaveBeenCalled();
  });
});
