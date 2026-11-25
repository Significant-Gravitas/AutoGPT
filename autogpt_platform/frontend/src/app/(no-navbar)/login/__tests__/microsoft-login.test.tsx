import type { ReactNode } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
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
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  mockSearchParams = new URLSearchParams();
});

// Holds the request that starts a provider sign-in until `finish`; every
// other request goes through.
function holdProviderRequest() {
  let finish = () => {};
  const settled = new Promise<Response>((resolve) => {
    finish = () => resolve(new Response(JSON.stringify({})));
  });
  const passThrough = globalThis.fetch;
  vi.spyOn(globalThis, "fetch").mockImplementation((input, init) =>
    String(input).includes("/api/auth/login/with-provider")
      ? settled
      : passThrough(input, init),
  );
  return { finish, settled };
}

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

  test("names each sign-in button once, its logo being decorative", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED", "true");

    render(<LoginPage />);

    // A logo with the provider's name as alt text would be read out twice.
    for (const name of ["Continue with Microsoft", "Continue with Google"]) {
      expect(screen.getByRole("button", { name })).toBeDefined();
    }
  });

  test("keeps Google unavailable while Microsoft sign-in is starting", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED", "true");
    const { finish, settled } = holdProviderRequest();

    try {
      render(<LoginPage />);
      fireEvent.click(
        screen.getByRole("button", { name: /continue with microsoft/i }),
      );

      // Two sign-ins at once would race their redirects.
      await waitFor(() =>
        expect(
          screen
            .getByRole("button", { name: /continue with google/i })
            .hasAttribute("disabled"),
        ).toBe(true),
      );
    } finally {
      finish();
      await settled;
    }
  });

  test("shows the waitlist answer when the sign-up isn't allowed", async () => {
    mockSearchParams = new URLSearchParams({
      provider: "microsoft",
      error:
        "This_email_address_is_not_allowed_to_register._Please_contact_support_for_assistance.",
    });

    render(<LoginPage />);

    expect(
      await screen.findByRole("dialog", { name: "Email Not Allowed" }),
    ).toBeDefined();
    expect(screen.queryByText(/didn't finish/)).toBeNull();
  });

  test("tells someone with an AutoGPT address to sign up with Google", () => {
    mockSearchParams = new URLSearchParams({
      provider: "microsoft",
      error: "team_email_requires_google",
    });

    render(<LoginPage />);

    expect(
      screen.getByText(
        "An AutoGPT team address signs up with Google. Use Continue with Google instead.",
      ),
    ).toBeDefined();
  });

  test("explains a refused Microsoft sign-in without saying whether an account uses the address", () => {
    mockSearchParams = new URLSearchParams({
      provider: "microsoft",
      error: "account_not_linked",
    });

    render(<LoginPage />);

    expect(
      screen.getByText(
        "We can't sign you in with that Microsoft account. If you already have an AutoGPT account, log in the way you usually do. If not, sign up with your email address. You can connect Microsoft 365 afterwards.",
      ),
    ).toBeDefined();
  });

  test("keeps Microsoft's words out of another provider's refusal", async () => {
    // Every provider's refused sign-in comes back here; only Microsoft's
    // says which provider it was.
    mockSearchParams = new URLSearchParams({ error: "access_denied" });

    render(<LoginPage />);

    expect(await screen.findByLabelText("Email")).toBeDefined();
    expect(screen.queryByText(/Microsoft/)).toBeNull();
    expect(screen.queryByText(/didn't finish/)).toBeNull();
  });

  test("explains another provider's refused sign-in in words of its own", async () => {
    // Where Google, GitHub or Discord lands when it refuses an address it
    // didn't verify, or one an account already uses.
    mockSearchParams = new URLSearchParams({ error: "account_not_linked" });

    render(<LoginPage />);

    expect(
      await screen.findByText(
        "We can't sign you in with that account. If you already have an AutoGPT account, log in the way you usually do. If not, sign up with your email address.",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/Microsoft/)).toBeNull();
  });

  test("tells a banned account it is banned, not that sign-in didn't finish", async () => {
    mockSearchParams = new URLSearchParams({
      provider: "microsoft",
      error: "BANNED_USER",
    });

    render(<LoginPage />);

    expect(
      await screen.findByText(
        "This account has been banned. If you think this is a mistake, please contact support.",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/didn't finish/)).toBeNull();
  });
});
