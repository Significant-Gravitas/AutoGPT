import type { ReactElement } from "react";
import { renderToString } from "react-dom/server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { SessionCookieProvider } from "@/components/auth/SessionCookieProvider";
import LoginPage from "../page";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockLoginAction = vi.hoisted(() => vi.fn());

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

// The server and the browser each have their own query client, as in the app.
function newQueryClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } });
}

function loginPage(hasSessionCookie: boolean, queryClient: QueryClient) {
  return (
    <QueryClientProvider client={queryClient}>
      <SessionCookieProvider hasSessionCookie={hasSessionCookie}>
        <LoginPage />
      </SessionCookieProvider>
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

  test("a visitor without a session cookie gets the form in the server HTML, not yet submittable", () => {
    showServerHTML(loginPage(false, newQueryClient()));

    expect(
      screen.getByRole("heading", {
        name: "Log in to your account to continue",
      }),
    ).toBeDefined();
    expect(screen.getByLabelText("Email")).toBeDefined();
    expect(logInButton().hasAttribute("disabled")).toBe(true);
  });

  test("logs in with what was typed before the page hydrated", async () => {
    const container = showServerHTML(loginPage(false, newQueryClient()));
    typeBeforeHydration(container, "email", "early@example.com");
    typeBeforeHydration(container, "password", "typed-before-js");
    const consoleError = vi.spyOn(console, "error");
    const browserQueryClient = newQueryClient();

    const { rerender } = render(loginPage(false, browserQueryClient), {
      container,
      hydrate: true,
    });
    mockUseAuth.mockReturnValue(SIGNED_OUT);
    rerender(loginPage(false, browserQueryClient));

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

  test("a visitor with a session cookie waits on the skeleton, and a signed-in one never sees the form", () => {
    const container = showServerHTML(loginPage(true, newQueryClient()));
    expect(screen.queryByLabelText("Email")).toBeNull();
    const browserQueryClient = newQueryClient();

    const { rerender } = render(loginPage(true, browserQueryClient), {
      container,
      hydrate: true,
    });
    mockUseAuth.mockReturnValue(SIGNED_IN);
    rerender(loginPage(true, browserQueryClient));

    expect(screen.queryByLabelText("Email")).toBeNull();
  });
});
