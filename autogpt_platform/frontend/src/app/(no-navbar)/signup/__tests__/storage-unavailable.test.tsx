import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { ReactNode } from "react";

const captureException = vi.hoisted(() => vi.fn());
const isServerSide = vi.hoisted(() => vi.fn(() => false));
const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@sentry/nextjs", () => ({ captureException }));

vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return { ...actual, environment: { ...actual.environment, isServerSide } };
});

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("../actions", () => ({ signup: vi.fn() }));

/**
 * `/signup` is one of the three routes the "Local storage is not available"
 * errors were filed against. The storage read comes from `OrgTeamProvider`,
 * which `src/app/providers.tsx` wraps every route in.
 */
describe("/signup when localStorage is unavailable", () => {
  const originalLocalStorage = Object.getOwnPropertyDescriptor(
    window,
    "localStorage",
  );

  beforeEach(() => {
    vi.resetModules();
    captureException.mockClear();
    isServerSide.mockReturnValue(false);
    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: false,
      isLoggedIn: false,
    });
  });

  afterEach(() => {
    if (originalLocalStorage) {
      Object.defineProperty(window, "localStorage", originalLocalStorage);
    }
  });

  test("server-rendering the route reports nothing to Sentry", async () => {
    isServerSide.mockReturnValue(true);
    const { renderToString } = await import("react-dom/server");
    const { default: OrgTeamProvider } = await import(
      "@/providers/org-team/OrgTeamProvider"
    );
    const { default: SignupPage } = await import("../page");

    const html = renderToString(
      <OrgTeamProvider>
        <SignupPage />
      </OrgTeamProvider>,
    );

    // /signup server-renders its loading shell until auth initialises in an
    // effect (#14281); the marketing panel is in both states.
    expect(html).toContain("Run in minutes");
    expect(captureException).not.toHaveBeenCalled();
  }, 20_000);

  test("renders in a browser that blocks storage, reporting nothing", async () => {
    Object.defineProperty(window, "localStorage", {
      get() {
        throw new Error("The operation is insecure.");
      },
      configurable: true,
    });
    const { default: OrgTeamProvider } = await import(
      "@/providers/org-team/OrgTeamProvider"
    );
    const { default: SignupPage } = await import("../page");

    render(
      <OrgTeamProvider>
        <SignupPage />
      </OrgTeamProvider>,
    );

    expect(await screen.findByLabelText("Email")).toBeDefined();
    expect(screen.getByRole("button", { name: "Sign up" })).toBeDefined();
    expect(captureException).not.toHaveBeenCalled();
  });
});
