import type { ReactNode } from "react";
import { http, HttpResponse } from "msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

const mocks = vi.hoisted(() => ({
  searchParams: new URLSearchParams(),
  replace: vi.fn(),
  toast: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: mocks.replace, push: vi.fn() }),
  useSearchParams: () => mocks.searchParams,
  usePathname: () => "/login",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toasts: [], toast: mocks.toast, dismiss: vi.fn() }),
}));

// Where a provider sign-in lands when it waits on an emailed code.
const WAITING =
  "error=sign_in_code_sent&error_description=sign_in_code_sent&provider=microsoft&next=%2Fmarketplace";

const originalLocation = window.location;
let pageLocation = { origin: "http://localhost", href: "http://localhost/" };

function answerStatus(pending: object | null) {
  server.use(
    http.get("*/api/auth/sign-in-code/status", () =>
      HttpResponse.json({ pending }),
    ),
  );
}

const PENDING = {
  provider: "microsoft",
  email: "pat@company.example",
  resendInSeconds: 60,
};

function openLogin(query: string) {
  mocks.searchParams = new URLSearchParams(query);
  render(<LoginPage />);
}

async function enterCode(code: string) {
  fireEvent.change(await screen.findByLabelText("Code"), {
    target: { value: code },
  });
  fireEvent.click(screen.getByRole("button", { name: "Continue" }));
}

beforeEach(() => {
  pageLocation = { origin: "http://localhost", href: "http://localhost/" };
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: pageLocation,
  });
});

afterEach(() => {
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: originalLocation,
  });
  mocks.replace.mockReset();
  mocks.toast.mockReset();
});

describe("LoginPage while a sign-in waits on its emailed code", () => {
  test("asks for the code, naming where it went and the provider", async () => {
    answerStatus(PENDING);
    openLogin(WAITING);

    expect(await screen.findByText("pat@company.example")).toBeDefined();
    expect(screen.getByText(/finish signing in with Microsoft/)).toBeDefined();
    expect(screen.queryByLabelText("Email")).toBeNull();
    expect(
      screen.queryByLabelText("Password", { selector: "input" }),
    ).toBeNull();
  });

  test("the right code continues where the sign-in was going", async () => {
    answerStatus(PENDING);
    let sent: unknown = null;
    server.use(
      http.post("*/api/auth/sign-in-code/verify", async ({ request }) => {
        sent = await request.json();
        return HttpResponse.json({ provider: "microsoft", newUser: false });
      }),
    );
    openLogin(WAITING);

    await enterCode("123 456");

    await waitFor(() =>
      expect(pageLocation.href).toBe(
        "/auth/callback?provider=microsoft&next=%2Fmarketplace",
      ),
    );
    // Sent as typed: the server ignores the space.
    expect(sent).toEqual({ code: "123 456" });
  });

  test("a wrong code says so and keeps what was typed", async () => {
    answerStatus(PENDING);
    server.use(
      http.post("*/api/auth/sign-in-code/verify", () =>
        HttpResponse.json(
          { code: "sign_in_code_invalid", message: "That code isn't right." },
          { status: 400 },
        ),
      ),
    );
    openLogin(WAITING);

    await enterCode("000000");

    expect(
      await screen.findByText(
        "That code isn't right. Check the email and try again.",
      ),
    ).toBeDefined();
    expect((screen.getByLabelText("Code") as HTMLInputElement).value).toBe(
      "000000",
    );
    expect(pageLocation.href).toBe("http://localhost/");
  });

  test("a sign-in that ran out offers the way back to the form", async () => {
    answerStatus(PENDING);
    server.use(
      http.post("*/api/auth/sign-in-code/verify", () =>
        HttpResponse.json(
          { code: "sign_in_code_expired", message: "Expired." },
          { status: 400 },
        ),
      ),
    );
    openLogin(WAITING);

    await enterCode("123456");
    expect(await screen.findByText("This sign-in has expired")).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Back to log in" }));

    expect(mocks.replace).toHaveBeenCalledWith("/login?next=%2Fmarketplace");
  });

  test("Wrong account? goes back to the form, keeping the deep link", async () => {
    answerStatus(PENDING);
    openLogin(WAITING);

    await screen.findByText("pat@company.example");
    fireEvent.click(screen.getByRole("button", { name: "Back to log in" }));

    expect(mocks.replace).toHaveBeenCalledWith("/login?next=%2Fmarketplace");
  });

  test("with nothing waiting in this browser, says the sign-in has expired", async () => {
    answerStatus(null);
    openLogin(WAITING);

    expect(await screen.findByText("This sign-in has expired")).toBeDefined();
  });

  test("still takes the code when looking the sign-in up fails", async () => {
    server.use(
      http.get(
        "*/api/auth/sign-in-code/status",
        () => new HttpResponse(null, { status: 429 }),
      ),
    );
    openLogin(WAITING);

    expect(
      await screen.findByText(/We sent a 6-digit code to your email address/),
    ).toBeDefined();
    expect(screen.getByLabelText("Code")).toBeDefined();
  });

  test("shows a refusal from the sign-up checks in the app's words", async () => {
    answerStatus(PENDING);
    server.use(
      http.post("*/api/auth/sign-in-code/verify", () =>
        HttpResponse.json(
          { code: "team_email_requires_google", message: "" },
          { status: 403 },
        ),
      ),
    );
    openLogin(WAITING);

    await enterCode("123456");

    expect(
      await screen.findByText(
        "An AutoGPT team address signs up with Google. Use Continue with Google instead.",
      ),
    ).toBeDefined();
  });

  test("sends a new code once the wait is over", async () => {
    answerStatus({ ...PENDING, resendInSeconds: 0 });
    let resent = false;
    server.use(
      http.post("*/api/auth/sign-in-code/resend", () => {
        resent = true;
        return HttpResponse.json({ resendInSeconds: 60 });
      }),
    );
    openLogin(WAITING);

    fireEvent.click(
      await screen.findByRole("button", { name: "Send a new code" }),
    );

    await waitFor(() =>
      expect(mocks.toast).toHaveBeenCalledWith({
        title: "New code sent to pat@company.example",
        variant: "success",
      }),
    );
    expect(resent).toBe(true);
    expect(
      screen.getByRole("button", { name: /Send a new code in \d+s/ }),
    ).toBeDefined();
  });

  test("holds the new code until a minute after the last one", async () => {
    answerStatus(PENDING);
    openLogin(WAITING);

    const resend = await screen.findByRole("button", {
      name: /Send a new code in \d+s/,
    });

    expect((resend as HTMLButtonElement).disabled).toBe(true);
  });
});

describe("LoginPage after a sign-in whose code couldn't go out", () => {
  test.each([
    [
      "sign_in_code_limit",
      "We've already sent several codes to this address. Try again in an hour, or log in the way you usually do.",
    ],
    [
      "sign_in_code_unsent",
      "We couldn't email you a code. Try again in a moment, or log in the way you usually do.",
    ],
  ])("explains %s with the form still there", async (code, copy) => {
    openLogin(`error=${code}&error_description=${code}`);

    expect(await screen.findByText(copy)).toBeDefined();
    expect(screen.getByLabelText("Email")).toBeDefined();
  });
});
