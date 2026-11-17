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
  vi.useRealTimers();
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

  test.each([
    ["sign_in_code_expired", 400, "This sign-in has expired"],
    ["sign_in_code_out_of_tries", 400, "Too many wrong codes"],
    ["sign_in_code_locked", 429, "Codes are paused for this address"],
    ["sign_in_code_unchecked", 503, "Something went wrong"],
  ])(
    "after %s, says why the sign-in ended and offers the way back to the form",
    async (code, status, title) => {
      answerStatus(PENDING);
      server.use(
        http.post("*/api/auth/sign-in-code/verify", () =>
          HttpResponse.json({ code, message: "" }, { status }),
        ),
      );
      openLogin(WAITING);

      await enterCode("123456");
      expect(await screen.findByRole("heading", { name: title })).toBeDefined();
      expect(screen.queryByLabelText("Code")).toBeNull();
      fireEvent.click(screen.getByRole("button", { name: "Back to log in" }));

      expect(mocks.replace).toHaveBeenCalledWith("/login?next=%2Fmarketplace");
    },
  );

  test("a new code refused because the address is locked ends the sign-in", async () => {
    answerStatus({ ...PENDING, resendInSeconds: 0 });
    server.use(
      http.post("*/api/auth/sign-in-code/resend", () =>
        HttpResponse.json(
          { code: "sign_in_code_locked", message: "" },
          { status: 429 },
        ),
      ),
    );
    openLogin(WAITING);

    fireEvent.click(
      await screen.findByRole("button", { name: "Send a new code" }),
    );

    expect(
      await screen.findByRole("heading", {
        name: "Codes are paused for this address",
      }),
    ).toBeDefined();
    expect(mocks.toast).not.toHaveBeenCalled();
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
      await screen.findByText(/You can ask for another in \d+s/),
    ).toBeDefined();
    expect(
      (
        screen.getByRole("button", {
          name: "Send a new code",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  test("counts time the tab slept toward the next code", async () => {
    answerStatus(PENDING);
    openLogin(WAITING);
    await screen.findByText(/You can ask for another in \d+s/);

    // The tab sleeps 90 seconds: the clock moves on, and no timer runs in
    // between.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(Date.now() + 90_000);

    await waitFor(() =>
      expect(screen.queryByText(/You can ask for another in/)).toBeNull(),
    );
    expect(
      (
        screen.getByRole("button", {
          name: "Send a new code",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(false);
  });

  test("holds the new code until a minute after the last one", async () => {
    // A still clock, so the countdown reads what the server said.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-17T12:30:00Z"));
    answerStatus(PENDING);
    openLogin(WAITING);

    expect(
      await screen.findByText("You can ask for another in 60s."),
    ).toBeDefined();
    // One name throughout, so a screen reader doesn't announce it each second.
    const resend = screen.getByRole("button", { name: "Send a new code" });
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
      "sign_in_code_locked",
      "Too many wrong codes were entered for this address, so we've stopped sending it codes for now. Log in the way you usually do.",
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
