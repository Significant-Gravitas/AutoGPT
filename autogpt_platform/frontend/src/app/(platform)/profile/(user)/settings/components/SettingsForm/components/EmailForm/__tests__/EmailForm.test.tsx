import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import type { User } from "@supabase/supabase-js";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { EmailForm } from "../EmailForm";

const mockToast = vi.hoisted(() => vi.fn());
const mockMutateAsync = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  usePostV1UpdateUserEmail: () => ({
    mutateAsync: mockMutateAsync,
    isPending: false,
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
} as User;

describe("EmailForm", () => {
  beforeEach(() => {
    mockToast.mockReset();
    mockMutateAsync.mockReset();
    mockMutateAsync.mockResolvedValue({});
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("submits a changed email to both update endpoints", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<EmailForm user={testUser} />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "updated@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Update email" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith("/api/auth/user", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email: "updated@example.com" }),
      });
    });
    await waitFor(() => {
      expect(mockMutateAsync).toHaveBeenCalledWith({
        data: "updated@example.com",
      });
    });
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Successfully updated email",
      }),
    );
  });

  test("keeps submit disabled when the email has not changed", () => {
    render(<EmailForm user={testUser} />);

    expect(
      (
        screen.getByRole("button", {
          name: "Update email",
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });
});
