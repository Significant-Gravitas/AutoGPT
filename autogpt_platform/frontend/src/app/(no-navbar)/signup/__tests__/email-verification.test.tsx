import type { ReactNode } from "react";
import { http, HttpResponse } from "msw";
import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { Toaster } from "@/components/molecules/Toast/toaster";
import SignupPage from "../page";

const mockSignupAction = vi.hoisted(() => vi.fn());
const routerReplace = vi.hoisted(() => vi.fn());

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: routerReplace, push: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/signup",
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: false }),
}));

vi.mock("../actions", () => ({
  signup: mockSignupAction,
}));

const email = "new.user@example.com";
const password = "validpassword123";

function submitSignupForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: email },
  });
  fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
    target: { value: password },
  });
  fireEvent.change(
    screen.getByLabelText("Confirm Password", { selector: "input" }),
    { target: { value: password } },
  );
  fireEvent.click(screen.getByRole("button", { name: "Sign up" }));
}

function verificationRequired() {
  mockSignupAction.mockResolvedValue({
    success: true,
    verificationRequired: true,
    email,
  });
}

async function waitOutCooldown() {
  // Each second re-arms its timer after a render, and shouldAdvanceTime moves
  // the fake clock in 20ms steps meanwhile, so the countdown can trail exactly
  // 60s by a step: tick until the button frees up instead.
  for (let second = 0; second < 65; second++) {
    if (screen.queryByRole("button", { name: "Resend email" })) return;
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });
  }
}

describe("SignupPage with email verification required", () => {
  beforeEach(() => {
    mockSignupAction.mockReset();
    routerReplace.mockReset();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  test("shows check-your-inbox with the address instead of leaving the page", async () => {
    verificationRequired();
    render(<SignupPage />);

    submitSignupForm();

    const heading = await screen.findByRole("heading", {
      name: "Check your inbox",
    });
    await waitFor(() => expect(document.activeElement).toBe(heading));
    expect(screen.getByText(email)).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Resend email in 60s" }),
    ).toHaveProperty("disabled", true);
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("resends the link through Better Auth once the cooldown is over", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const resendBodies: unknown[] = [];
    server.use(
      http.post("*/api/auth/send-verification-email", async ({ request }) => {
        resendBodies.push(await request.json());
        return HttpResponse.json({ status: true });
      }),
    );
    verificationRequired();
    render(<SignupPage />);

    submitSignupForm();
    await screen.findByRole("heading", { name: "Check your inbox" });
    await waitOutCooldown();
    fireEvent.click(screen.getByRole("button", { name: "Resend email" }));

    await waitFor(() => {
      expect(resendBodies).toEqual([
        { email, callbackURL: "/auth/callback?method=email" },
      ]);
    });
    expect(
      await screen.findByRole("button", { name: "Resend email in 60s" }),
    ).toHaveProperty("disabled", true);
  });

  test("a resent link still carries the sign-up's marketing opt-out", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const resendBodies: unknown[] = [];
    server.use(
      http.post("*/api/auth/send-verification-email", async ({ request }) => {
        resendBodies.push(await request.json());
        return HttpResponse.json({ status: true });
      }),
    );
    verificationRequired();
    render(<SignupPage />);

    fireEvent.click(screen.getByRole("button", { name: "opt out" }));
    submitSignupForm();
    await screen.findByRole("heading", { name: "Check your inbox" });
    await waitOutCooldown();
    fireEvent.click(screen.getByRole("button", { name: "Resend email" }));

    await waitFor(() => {
      expect(resendBodies).toEqual([
        {
          email,
          callbackURL: "/auth/callback?method=email&marketing_opt_out=1",
        },
      ]);
    });
  });

  test("keeps the button available when the resend fails", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    server.use(
      http.post("*/api/auth/send-verification-email", () =>
        HttpResponse.json({ message: "Too many requests" }, { status: 429 }),
      ),
    );
    verificationRequired();
    render(<SignupPage />);

    submitSignupForm();
    await screen.findByRole("heading", { name: "Check your inbox" });
    await waitOutCooldown();
    fireEvent.click(screen.getByRole("button", { name: "Resend email" }));

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Resend email" }),
      ).toHaveProperty("disabled", false);
    });
  });

  test("shows an error and frees the button when the resend can't reach the server", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let networkUp = false;
    server.use(
      http.post("*/api/auth/send-verification-email", () =>
        networkUp ? HttpResponse.json({ status: true }) : HttpResponse.error(),
      ),
    );
    verificationRequired();
    render(
      <>
        <SignupPage />
        <Toaster />
      </>,
    );

    submitSignupForm();
    await screen.findByRole("heading", { name: "Check your inbox" });
    await waitOutCooldown();
    fireEvent.click(screen.getByRole("button", { name: "Resend email" }));

    expect(await screen.findByText("We couldn't send the email")).toBeDefined();
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Resend email" }),
      ).toHaveProperty("disabled", false);
    });

    networkUp = true;
    fireEvent.click(screen.getByRole("button", { name: "Resend email" }));

    expect(
      await screen.findByRole("button", { name: "Resend email in 60s" }),
    ).toHaveProperty("disabled", true);
    expect(await screen.findByText(`Email sent to ${email}`)).toBeDefined();
  });

  test("start again returns to the form with the address cleared", async () => {
    verificationRequired();
    render(<SignupPage />);

    submitSignupForm();
    await screen.findByRole("heading", { name: "Check your inbox" });
    fireEvent.click(screen.getByRole("button", { name: "Start again" }));

    const emailInput = (await screen.findByLabelText(
      "Email",
    )) as HTMLInputElement;
    expect(emailInput.value).toBe("");
    expect(screen.getByRole("button", { name: "Sign up" })).toBeDefined();
  });

  test("with verification off, a sign-up with a session still goes straight in", async () => {
    mockSignupAction.mockResolvedValue({ success: true, next: "/onboarding" });
    render(<SignupPage />);

    submitSignupForm();

    await waitFor(() => {
      expect(routerReplace).toHaveBeenCalledWith("/onboarding");
    });
    expect(
      screen.queryByRole("heading", { name: "Check your inbox" }),
    ).toBeNull();
  });
});
