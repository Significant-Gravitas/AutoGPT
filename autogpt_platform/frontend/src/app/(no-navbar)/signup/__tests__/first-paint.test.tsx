import type { ReactElement } from "react";
import { renderToString } from "react-dom/server";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { SessionCookieProvider } from "@/components/auth/SessionCookieProvider";
import SignupPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/signup",
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("../actions", () => ({ signup: vi.fn() }));

// The auth store's state on the server and until its session check answers.
const CHECKING = { user: null, isUserLoading: true, isLoggedIn: false };
const SIGNED_OUT = { user: null, isUserLoading: false, isLoggedIn: false };
const SIGNED_IN = { user: { id: "u" }, isUserLoading: false, isLoggedIn: true };

function signupPage(hasSessionCookie: boolean) {
  return (
    <SessionCookieProvider hasSessionCookie={hasSessionCookie}>
      <SignupPage />
    </SessionCookieProvider>
  );
}

const serverContainers: HTMLElement[] = [];

// The server's HTML in the document, as a browser shows it before the app's
// JavaScript has run.
function showServerHTML(ui: ReactElement) {
  const container = document.createElement("div");
  container.innerHTML = renderToString(ui);
  document.body.appendChild(container);
  serverContainers.push(container);
  return container;
}

function signUpButton() {
  return screen.getByRole("button", { name: "Sign up" });
}

describe("SignupPage first paint", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue(CHECKING);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const container of serverContainers.splice(0)) container.remove();
  });

  test("a visitor without a session cookie gets the form in the server HTML, not yet submittable", () => {
    showServerHTML(signupPage(false));

    expect(
      screen.getByRole("heading", { name: "Create your account" }),
    ).toBeDefined();
    expect(screen.getByLabelText("Email")).toBeDefined();
    expect(signUpButton().hasAttribute("disabled")).toBe(true);
  });

  test("keeps what was typed before the page hydrated, and enables Sign up once it has", async () => {
    const container = showServerHTML(signupPage(false));
    const email = container.querySelector<HTMLInputElement>("#email");
    if (!email) throw new Error("the server HTML has no email field");
    email.value = "early@example.com";
    const consoleError = vi.spyOn(console, "error");

    const { rerender } = render(signupPage(false), {
      container,
      hydrate: true,
    });
    mockUseAuth.mockReturnValue(SIGNED_OUT);
    rerender(signupPage(false));

    await waitFor(() =>
      expect(signUpButton().hasAttribute("disabled")).toBe(false),
    );
    expect((screen.getByLabelText("Email") as HTMLInputElement).value).toBe(
      "early@example.com",
    );
    expect(consoleError.mock.calls.flat().join(" ")).not.toMatch(/hydrat/i);
  });

  test("a visitor with a session cookie waits on the skeleton, and a signed-in one never sees the form", () => {
    const container = showServerHTML(signupPage(true));
    expect(screen.queryByLabelText("Email")).toBeNull();

    const { rerender } = render(signupPage(true), { container, hydrate: true });
    mockUseAuth.mockReturnValue(SIGNED_IN);
    rerender(signupPage(true));

    expect(screen.queryByLabelText("Email")).toBeNull();
  });

  test("a visitor whose session cookie no longer signs them in gets the form after the check", async () => {
    const container = showServerHTML(signupPage(true));

    const { rerender } = render(signupPage(true), { container, hydrate: true });
    mockUseAuth.mockReturnValue(SIGNED_OUT);
    rerender(signupPage(true));

    expect(await screen.findByLabelText("Email")).toBeDefined();
  });
});
