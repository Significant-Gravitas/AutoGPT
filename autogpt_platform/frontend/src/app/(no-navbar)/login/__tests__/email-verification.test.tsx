import type { ReactNode } from "react";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { beforeEach, describe, expect, test, vi } from "vitest";
import LoginPage from "../page";

const mockLoginAction = vi.hoisted(() => vi.fn());
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

vi.mock("../actions", () => ({
  login: mockLoginAction,
}));

const email = "unverified@example.com";

function submitLoginForm() {
  fireEvent.change(screen.getByLabelText("Email"), {
    target: { value: email },
  });
  fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
    target: { value: "hunter2-password" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Log in" }));
}

describe("LoginPage for an unverified email", () => {
  beforeEach(() => {
    mockLoginAction.mockReset();
    mockSearchParams = new URLSearchParams();
  });

  test("shows check-your-inbox instead of a dead-end error", async () => {
    mockLoginAction.mockResolvedValue({
      success: false,
      error: "email_not_verified",
      email,
    });
    render(<LoginPage />);

    submitLoginForm();

    expect(
      await screen.findByRole("heading", {
        name: "Verify your email to log in",
      }),
    ).toBeDefined();
    expect(screen.getByText(email)).toBeDefined();
    expect(mockLoginAction).toHaveBeenCalledWith(
      email,
      "hunter2-password",
      null,
    );
  });

  test("back to log in returns to the form", async () => {
    mockLoginAction.mockResolvedValue({
      success: false,
      error: "email_not_verified",
      email,
    });
    render(<LoginPage />);

    submitLoginForm();
    fireEvent.click(
      await screen.findByRole("button", { name: "Back to log in" }),
    );

    expect(await screen.findByRole("button", { name: "Log in" })).toBeDefined();
  });

  test("explains an expired or used verification link", () => {
    mockSearchParams = new URLSearchParams({ email_verification: "expired" });
    render(<LoginPage />);

    expect(
      screen.getByText(
        "That verification link has expired or was already used. Log in and we'll email you a new one.",
      ),
    ).toBeDefined();
  });

  test("ignores an unknown notice value", () => {
    mockSearchParams = new URLSearchParams({ email_verification: "<b>x</b>" });
    render(<LoginPage />);

    expect(screen.queryByRole("alert")).toBeNull();
  });
});
