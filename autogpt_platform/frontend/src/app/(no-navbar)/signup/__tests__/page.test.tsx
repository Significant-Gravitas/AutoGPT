import type { ReactNode } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import SignupPage from "../page";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseAuth = vi.hoisted(() => vi.fn());
const mockSignupAction = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("../actions", () => ({
  signup: mockSignupAction,
}));

describe("SignupPage", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: false,
      isLoggedIn: false,
    });
    mockSignupAction.mockReset();
  });

  test("shows existing user feedback from signup action", async () => {
    mockSignupAction.mockResolvedValue({
      success: false,
      error: "user_already_exists",
    });

    render(<SignupPage />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "existing@example.com" },
    });
    fireEvent.change(screen.getByLabelText("Password", { selector: "input" }), {
      target: { value: "validpassword123" },
    });
    fireEvent.change(
      screen.getByLabelText("Confirm Password", { selector: "input" }),
      {
        target: { value: "validpassword123" },
      },
    );
    fireEvent.click(screen.getByRole("checkbox"));
    fireEvent.click(screen.getByRole("button", { name: "Sign up" }));

    await waitFor(() => {
      expect(mockSignupAction).toHaveBeenCalledWith(
        "existing@example.com",
        "validpassword123",
        "validpassword123",
        true,
      );
    });

    expect(
      await screen.findByText("User with this email already exists"),
    ).toBeDefined();
  });

  test("does not server-render an interactive form before auth initializes", () => {
    const markup = renderToStaticMarkup(<SignupPage />);

    expect(markup).not.toContain('id="password"');
    expect(markup).not.toContain('type="submit"');
  });

  test("preserves form input during a background auth refresh", () => {
    const { rerender } = render(<SignupPage />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "draft@example.com" },
    });

    mockUseAuth.mockReturnValue({
      user: null,
      isUserLoading: true,
      isLoggedIn: false,
    });
    rerender(<SignupPage />);

    expect((screen.getByLabelText("Email") as HTMLInputElement).value).toBe(
      "draft@example.com",
    );
  });
});
