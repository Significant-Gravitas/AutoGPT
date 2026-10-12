import type { ReactNode } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { IMPERSONATION_COOKIE_NAME } from "@/lib/constants";
import { ImpersonationState } from "@/lib/impersonation";
import LoginPage from "../page";

const mockLoginAction = vi.hoisted(() => vi.fn());
const mockToast = vi.hoisted(() => vi.fn());

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

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useToast: () => ({ toasts: [], toast: mockToast, dismiss: vi.fn() }),
}));

const TEAM_SSO_TOAST = "Please use Google SSO to login using an AutoGPT email.";

function submitLoginForm(email: string) {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: email },
  });
  fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
    target: { value: "hunter2-password" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Log in" }));
}

function impersonationCookiePresent() {
  return document.cookie.includes(`${IMPERSONATION_COOKIE_NAME}=`);
}

describe("LoginPage sign-in guards", () => {
  beforeEach(() => {
    mockLoginAction.mockReset();
    mockLoginAction.mockResolvedValue({ success: false, error: "nope" });
    mockToast.mockReset();
  });

  afterEach(() => {
    ImpersonationState.clear();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  test.each(["someone@AGPT.CO", "someone@agpt.co"])(
    "sends the team address %s to Google SSO instead of password login",
    async (email) => {
      render(<LoginPage />);

      submitLoginForm(email);

      await waitFor(() =>
        expect(mockToast).toHaveBeenCalledWith(
          expect.objectContaining({ title: TEAM_SSO_TOAST }),
        ),
      );
      expect(mockLoginAction).not.toHaveBeenCalled();
    },
  );

  test.each(["someone@agpt.com", "someone@agpt.co.uk", "qa@previews.agpt.co"])(
    "lets the look-alike address %s use password login",
    async (email) => {
      render(<LoginPage />);

      submitLoginForm(email);

      await waitFor(() =>
        expect(mockLoginAction).toHaveBeenCalledWith(
          email,
          "hunter2-password",
          null,
        ),
      );
      expect(mockToast).not.toHaveBeenCalledWith(
        expect.objectContaining({ title: TEAM_SSO_TOAST }),
      );
    },
  );

  test("drops leftover admin impersonation before a password sign-in", async () => {
    ImpersonationState.set("someone-elses-user-id", "admin-1");
    expect(impersonationCookiePresent()).toBe(true);
    let impersonationAtLogin: string | null = "unset";
    mockLoginAction.mockImplementation(async () => {
      impersonationAtLogin = ImpersonationState.get();
      return { success: false, error: "nope" };
    });
    render(<LoginPage />);

    submitLoginForm("member@example.com");

    await waitFor(() => expect(mockLoginAction).toHaveBeenCalledTimes(1));
    expect(impersonationAtLogin).toBeNull();
    expect(impersonationCookiePresent()).toBe(false);
  });

  test("drops leftover admin impersonation before a Google sign-in", async () => {
    ImpersonationState.set("someone-elses-user-id", "admin-1");
    expect(impersonationCookiePresent()).toBe(true);
    let stateAtProviderRequest: { session: string | null; cookie: boolean } = {
      session: "unset",
      cookie: true,
    };
    const fetchSpy = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async () => {
        stateAtProviderRequest = {
          session: ImpersonationState.get(),
          cookie: impersonationCookiePresent(),
        };
        return new Response(JSON.stringify({}));
      });
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    render(<LoginPage />);

    fireEvent.click(
      screen.getByRole("button", { name: /continue with google/i }),
    );

    await waitFor(() => expect(fetchSpy).toHaveBeenCalled());
    expect(stateAtProviderRequest).toEqual({ session: null, cookie: false });
  });
});
