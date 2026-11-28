import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

vi.mock("../actions", () => ({ login: vi.fn() }));

afterEach(() => vi.unstubAllEnvs());

describe("Login terms notice", () => {
  test("shows the signup terms beneath the provider buttons when Google login is available", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    vi.stubEnv("NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED", "true");
    render(<LoginPage />);

    const terms = screen.getByRole("link", { name: "Terms of Use" });
    const privacy = screen.getByRole("link", { name: "Privacy Policy" });
    expect(terms.getAttribute("href")).toBe(
      "https://agpt.co/legal/platform-terms-of-use",
    );
    expect(privacy.getAttribute("href")).toBe(
      "https://agpt.co/legal/platform-privacy-policy",
    );
    for (const link of [terms, privacy]) {
      expect(link.getAttribute("target")).toBe("_blank");
      expect(link.getAttribute("rel")).toBe("noopener noreferrer");
    }
    const notice = screen.getByText(/By continuing you agree to our/);
    expect(notice.textContent).toBe(
      "By continuing you agree to our Terms of Use and Privacy Policy.",
    );
    for (const name of ["Continue with Google", "Continue with Microsoft"]) {
      expect(
        notice.previousElementSibling?.contains(
          screen.getByRole("button", { name }),
        ),
      ).toBe(true);
    }
  });

  test("keeps the notice off a login screen without Google", () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "LOCAL");
    vi.stubEnv("NEXT_PUBLIC_AUTH_MICROSOFT_ENABLED", "true");
    render(<LoginPage />);

    expect(
      screen.getByRole("button", { name: "Continue with Microsoft" }),
    ).toBeDefined();
    expect(
      screen.queryByRole("button", { name: "Continue with Google" }),
    ).toBeNull();
    expect(screen.queryByRole("link", { name: "Terms of Use" })).toBeNull();
    expect(screen.queryByRole("link", { name: "Privacy Policy" })).toBeNull();
  });
});
