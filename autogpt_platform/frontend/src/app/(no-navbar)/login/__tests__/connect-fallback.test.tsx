import type { ReactNode } from "react";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { rememberConnectAttempt } from "@/lib/auth/connect-attempt";
import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
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

// The provider sign-in requests the page sends, answered without a redirect.
function providerRequests(answer: Record<string, unknown> = {}) {
  const bodies: Array<Record<string, unknown>> = [];
  const passThrough = globalThis.fetch;
  vi.spyOn(globalThis, "fetch").mockImplementation((input, init) => {
    if (!String(input).includes("/api/auth/login/with-provider")) {
      return passThrough(input, init);
    }
    bodies.push(JSON.parse(String(init?.body)));
    return Promise.resolve(new Response(JSON.stringify(answer)));
  });
  return bodies;
}

beforeEach(() => {
  vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
  vi.stubEnv("NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED", "true");
  window.sessionStorage.clear();
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  mockSearchParams = new URLSearchParams();
  window.sessionStorage.clear();
});

const WITHOUT_GOOGLE = "Continue with Google without connecting";

// What Back fires when the browser brings the page back from its cache, as
// it was left.

describe("Signing in without connecting", () => {
  test("is offered after a connecting sign-in comes back refused", async () => {
    mockSearchParams = new URLSearchParams(
      "error=access_denied&connect=google",
    );
    const requests = providerRequests();

    render(<LoginPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: WITHOUT_GOOGLE }),
    );

    await waitFor(() => expect(requests).toHaveLength(1));
    expect(requests[0]).toMatchObject({ provider: "google", connect: false });
  });

  test("is offered after a connecting sign-in never came back", async () => {
    // A Google Workspace admin's block ends on Google's own error page.
    rememberConnectAttempt("microsoft");

    render(<LoginPage />);

    expect(
      await screen.findByRole("button", {
        name: "Continue with Microsoft without connecting",
      }),
    ).toBeDefined();
  });

  test("is offered when Back brings the page back as the sign-in left it", async () => {
    providerRequests({ url: "", connecting: true });
    render(<LoginPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Continue with Google" }),
    );
    await waitFor(() =>
      expect(
        window.sessionStorage.getItem("autogpt:connect-attempt"),
      ).not.toBeNull(),
    );

    restoreFromBackForwardCache();

    // Offered, and not held back by the sign-in the page was still starting.
    const fallback = await screen.findByRole("button", {
      name: WITHOUT_GOOGLE,
    });
    await waitFor(() => expect(fallback).toHaveProperty("disabled", false));
  });

  test("isn't offered on an ordinary visit", async () => {
    render(<LoginPage />);

    await screen.findByRole("button", { name: "Continue with Google" });
    expect(screen.queryByRole("button", { name: WITHOUT_GOOGLE })).toBeNull();
  });

  test("notes a connecting sign-in as it starts", async () => {
    providerRequests({ url: "", connecting: true });

    render(<LoginPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Continue with Google" }),
    );

    await waitFor(() =>
      expect(
        window.sessionStorage.getItem("autogpt:connect-attempt"),
      ).toContain('"provider":"google"'),
    );
  });

  test("forgets an attempt after a while", async () => {
    window.sessionStorage.setItem(
      "autogpt:connect-attempt",
      JSON.stringify({ provider: "google", at: Date.now() - 11 * 60 * 1000 }),
    );

    render(<LoginPage />);

    await screen.findByRole("button", { name: "Continue with Google" });
    expect(screen.queryByRole("button", { name: WITHOUT_GOOGLE })).toBeNull();
  });
});
