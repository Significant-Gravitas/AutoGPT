import type { ReactNode } from "react";
import {
  render,
  screen,
  fireEvent,
  waitFor,
} from "@/tests/integrations/test-utils";
import SignupPage from "../page";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseSupabase = vi.hoisted(() => vi.fn());
const mockSignupAction = vi.hoisted(() => vi.fn());

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

vi.mock("../actions", () => ({
  signup: mockSignupAction,
}));

describe("SignupPage", () => {
  beforeEach(() => {
    mockUseSupabase.mockReturnValue({
      supabase: {},
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
});
