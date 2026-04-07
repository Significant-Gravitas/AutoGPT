import { act, renderHook } from "@testing-library/react";
import type { User } from "@supabase/supabase-js";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { useEmailForm } from "./useEmailForm";

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

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
} as User;

describe("useEmailForm", () => {
  beforeEach(() => {
    mockToast.mockReset();
    mockMutateAsync.mockReset();
    mockMutateAsync.mockResolvedValue({});
  });

  test("submits a changed email to both update endpoints", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({}),
    });

    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useEmailForm({ user: testUser }));

    await act(async () => {
      await result.current.onSubmit({ email: "updated@example.com" });
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/auth/user", {
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email: "updated@example.com" }),
    });
    expect(mockMutateAsync).toHaveBeenCalledWith({
      data: "updated@example.com",
    });
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Successfully updated email",
      }),
    );

    vi.unstubAllGlobals();
  });

  test("skips the update when the email has not changed", async () => {
    const fetchMock = vi.fn();

    vi.stubGlobal("fetch", fetchMock);

    const { result } = renderHook(() => useEmailForm({ user: testUser }));

    await act(async () => {
      await result.current.onSubmit({ email: "user@example.com" });
    });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(mockMutateAsync).not.toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();

    vi.unstubAllGlobals();
  });
});
