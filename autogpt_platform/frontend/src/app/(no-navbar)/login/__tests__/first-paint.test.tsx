import type { ReactElement } from "react";
import { renderToString } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import LoginLayout from "../layout";
import LoginPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockLoginAction = vi.hoisted(() => vi.fn());
const mockHeaders = vi.hoisted(() => vi.fn());

vi.mock("next/headers", () => ({ headers: mockHeaders }));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/login",
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({ useAuth: mockUseAuth }));

vi.mock("../actions", () => ({ login: mockLoginAction }));

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<object>()),
  useToast: () => ({ toasts: [], toast: vi.fn(), dismiss: vi.fn() }),
}));

// The auth store's state on the server and until its session check answers.
const CHECKING = { user: null, isUserLoading: true, isLoggedIn: false };
const SIGNED_OUT = { user: null, isUserLoading: false, isLoggedIn: false };
const SIGNED_IN = { user: { id: "u" }, isUserLoading: false, isLoggedIn: true };

const OTHER_COOKIES = "CookieConsent=granted; ph_session=abc";
// Over plain http (local and self-hosted) the session cookie has no prefix.
const SESSION_COOKIE = `${OTHER_COOKIES}; better-auth.session_token=token.signature`;

// The server and the browser each have their own query client, as in the app.
function newQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

// The page as the server renders it for a request carrying these cookies.
async function loginPage(cookie: string, queryClient: QueryClient) {
  mockHeaders.mockResolvedValue(new Headers({ cookie }));
  return (
    <QueryClientProvider client={queryClient}>
      {await LoginLayout({ children: <LoginPage /> })}
    </QueryClientProvider>
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

function logInButton() {
  return screen.getByRole("button", { name: "Log in" });
}

function typeBeforeHydration(container: HTMLElement, id: string, text: string) {
  const input = container.querySelector<HTMLInputElement>(`#${id}`);
  if (!input) throw new Error(`the server HTML has no #${id} field`);
  input.value = text;
}

describe("LoginPage first paint", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue(CHECKING);
    mockLoginAction.mockReset();
    mockLoginAction.mockResolvedValue({ success: false, error: "nope" });
  });

  afterEach(() => {
    vi.restoreAllMocks();
    for (const container of serverContainers.splice(0)) container.remove();
  });

  test("a visitor without a session cookie gets the form in the server HTML, not yet submittable", async () => {
    showServerHTML(await loginPage(OTHER_COOKIES, newQueryClient()));

    expect(
      screen.getByRole("heading", {
        name: "Log in to your account to continue",
      }),
    ).toBeDefined();
    expect(screen.getByLabelText("Email")).toBeDefined();
    expect(logInButton().hasAttribute("disabled")).toBe(true);
  });

  test("logs in with what was typed before the page hydrated", async () => {
    const container = showServerHTML(
      await loginPage(OTHER_COOKIES, newQueryClient()),
    );
    typeBeforeHydration(container, "email", "early@example.com");
    typeBeforeHydration(container, "password", "typed-before-js");
    const consoleError = vi.spyOn(console, "error");
    const browserQueryClient = newQueryClient();

    const { rerender } = render(
      await loginPage(OTHER_COOKIES, browserQueryClient),
      { container, hydrate: true },
    );
    mockUseAuth.mockReturnValue(SIGNED_OUT);
    rerender(await loginPage(OTHER_COOKIES, browserQueryClient));

    await waitFor(() =>
      expect(logInButton().hasAttribute("disabled")).toBe(false),
    );
    fireEvent.click(logInButton());

    await waitFor(() =>
      expect(mockLoginAction).toHaveBeenCalledWith(
        "early@example.com",
        "typed-before-js",
        null,
      ),
    );
    expect(consoleError.mock.calls.flat().join(" ")).not.toMatch(/hydrat/i);
  });

  test("a visitor with a session cookie waits on the skeleton, and a signed-in one never sees the form", async () => {
    const container = showServerHTML(
      await loginPage(SESSION_COOKIE, newQueryClient()),
    );
    expect(screen.queryByLabelText("Email")).toBeNull();
    const browserQueryClient = newQueryClient();

    const { rerender } = render(
      await loginPage(SESSION_COOKIE, browserQueryClient),
      { container, hydrate: true },
    );
    mockUseAuth.mockReturnValue(SIGNED_IN);
    rerender(await loginPage(SESSION_COOKIE, browserQueryClient));

    expect(screen.queryByLabelText("Email")).toBeNull();
  });
});
