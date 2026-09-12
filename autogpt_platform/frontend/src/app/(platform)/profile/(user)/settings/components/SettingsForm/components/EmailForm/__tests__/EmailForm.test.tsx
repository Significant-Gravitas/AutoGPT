import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { ReactNode } from "react";
import type { User } from "@/lib/auth/types";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { EmailForm } from "../EmailForm";

const mockToast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

// Verified by default: the Supabase copy migration carries email_confirmed_at
// across, so users who had confirmed take this path. Unverified ones (never
// confirmed, or signed up since — signup verification is off) take the other.
const testUser = {
  id: "user-1",
  email: "user@example.com",
  email_verified: true,
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
} as User;

const unverifiedUser = { ...testUser, email_verified: false } as User;

describe("EmailForm", () => {
  beforeEach(() => {
    mockToast.mockReset();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  test("routes the change only through Better Auth (verification-gated), never an eager platform write", async () => {
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

    // The Better Auth PUT is the ONLY write — the old parallel platform-email
    // mutation is gone, so the change can't land unverified.
    expect(fetchMock).toHaveBeenCalledTimes(1);

    // Toast reflects the pending confirmation, not a completed change.
    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Confirm your new email",
        }),
      );
    });
  });

  test("tells an unverified user the change already applied", async () => {
    // Better Auth only sends the current-address confirmation when the user is
    // verified; for an unverified user it applies the change immediately, so
    // promising them a confirmation email would be a lie.
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<EmailForm user={unverifiedUser} />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "updated@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Update email" }));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Email updated",
        }),
      );
    });
  });

  test("surfaces a change failure as an error toast", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      json: async () => ({ error: "Email already in use" }),
    });

    vi.stubGlobal("fetch", fetchMock);

    render(<EmailForm user={testUser} />);

    fireEvent.change(screen.getByLabelText("Email"), {
      target: { value: "taken@example.com" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Update email" }));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Error updating email",
          description: "Email already in use",
          variant: "destructive",
        }),
      );
    });
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

  test("does not re-send after a successful request until the address changes again", async () => {
    // On the verified path Better Auth doesn't write the row until the link is
    // clicked, so user.email stays old. Without resetting the form the submit
    // button stays enabled and every further click sends another email.
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
      expect(fetchMock).toHaveBeenCalledTimes(1);
    });

    // The field is back to the current address, so the guard now short-circuits.
    await waitFor(() => {
      expect((screen.getByLabelText("Email") as HTMLInputElement).value).toBe(
        testUser.email,
      );
    });

    fireEvent.click(screen.getByRole("button", { name: "Update email" }));
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });
});
