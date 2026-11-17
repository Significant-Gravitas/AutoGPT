import type { ReactElement } from "react";
import { renderToString } from "react-dom/server";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import SignupLayout from "../layout";
import SignupPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockHeaders = vi.hoisted(() => vi.fn());

vi.mock("next/headers", () => ({ headers: mockHeaders }));

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

const OTHER_COOKIES = "CookieConsent=granted; ph_session=abc";
// Over https Better Auth names its session cookie with the __Secure- prefix.
const SESSION_COOKIE = `${OTHER_COOKIES}; __Secure-better-auth.session_token=token.signature`;

// The page as the server renders it for a request carrying these cookies.
async function signupPage(cookie: string) {
  mockHeaders.mockResolvedValue(new Headers({ cookie }));
  return SignupLayout({ children: <SignupPage /> });
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

  test("a visitor without a session cookie gets the form in the server HTML, not yet submittable", async () => {
    showServerHTML(await signupPage(OTHER_COOKIES));

    expect(
      screen.getByRole("heading", { name: "Create your account" }),
    ).toBeDefined();
    expect(screen.getByLabelText("Email")).toBeDefined();
    expect(signUpButton().hasAttribute("disabled")).toBe(true);
  });

  test("keeps what was typed before the page hydrated, and enables Sign up once it has", async () => {
    const page = await signupPage(OTHER_COOKIES);
    const container = showServerHTML(page);
    const email = container.querySelector<HTMLInputElement>("#email");
    if (!email) throw new Error("the server HTML has no email field");
    email.value = "early@example.com";
    const consoleError = vi.spyOn(console, "error");

    const { rerender } = render(page, { container, hydrate: true });
    mockUseAuth.mockReturnValue(SIGNED_OUT);
    rerender(await signupPage(OTHER_COOKIES));

    await waitFor(() =>
      expect(signUpButton().hasAttribute("disabled")).toBe(false),
    );
    expect((screen.getByLabelText("Email") as HTMLInputElement).value).toBe(
      "early@example.com",
    );
    expect(consoleError.mock.calls.flat().join(" ")).not.toMatch(/hydrat/i);
  });

  test("a visitor with a session cookie waits on the skeleton, and a signed-in one never sees the form", async () => {
    const page = await signupPage(SESSION_COOKIE);
    const container = showServerHTML(page);
    expect(screen.queryByLabelText("Email")).toBeNull();

    const { rerender } = render(page, { container, hydrate: true });
    mockUseAuth.mockReturnValue(SIGNED_IN);
    rerender(await signupPage(SESSION_COOKIE));

    expect(screen.queryByLabelText("Email")).toBeNull();
  });

  test("a visitor whose session cookie no longer signs them in gets the form after the check", async () => {
    const page = await signupPage(SESSION_COOKIE);
    const container = showServerHTML(page);

    const { rerender } = render(page, { container, hydrate: true });
    mockUseAuth.mockReturnValue(SIGNED_OUT);
    rerender(await signupPage(SESSION_COOKIE));

    expect(await screen.findByLabelText("Email")).toBeDefined();
  });
});
