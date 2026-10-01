import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

type MutationCallbacks = {
  onSuccess: (response: { status: number; data: unknown }) => void;
  onError: (error: unknown) => void;
};

const hoisted = vi.hoisted(() => ({
  toastSpy: vi.fn(),
  mutateSpy: vi.fn(),
  captured: { mutation: undefined as MutationCallbacks | undefined },
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: hoisted.toastSpy }),
  toast: hoisted.toastSpy,
}));

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  usePostV2GenerateTestData: (opts: { mutation: MutationCallbacks }) => {
    hoisted.captured.mutation = opts.mutation;
    return { mutate: hoisted.mutateSpy, isPending: false };
  },
}));

import { useGenerateTestDataButton } from "./useGenerateTestDataButton";

beforeEach(() => {
  hoisted.toastSpy.mockReset();
  hoisted.mutateSpy.mockReset();
  hoisted.captured.mutation = undefined;
});

describe("useGenerateTestDataButton", () => {
  test("submits the default e2e script type", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() => result.current.generate());
    expect(hoisted.mutateSpy).toHaveBeenCalledWith({
      data: { script_type: "e2e" },
    });
  });

  test("submits the selected full script type", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() => result.current.setScriptType("full"));
    act(() => result.current.generate());
    expect(hoisted.mutateSpy).toHaveBeenCalledWith({
      data: { script_type: "full" },
    });
  });

  test("stores the result and toasts success on a successful response", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() =>
      hoisted.captured.mutation?.onSuccess({
        status: 200,
        data: { success: true, message: "done", details: { users_created: 7 } },
      }),
    );
    expect(result.current.result?.message).toBe("done");
    expect(hoisted.toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Success" }),
    );
  });

  test("toasts a destructive error when the API reports failure", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() =>
      hoisted.captured.mutation?.onSuccess({
        status: 200,
        data: { success: false, message: "nope" },
      }),
    );
    expect(result.current.result?.success).toBe(false);
    expect(hoisted.toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Error", variant: "destructive" }),
    );
  });

  test("ignores non-200 success responses", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() =>
      hoisted.captured.mutation?.onSuccess({
        status: 403,
        data: { detail: "forbidden" },
      }),
    );
    expect(result.current.result).toBeNull();
    expect(hoisted.toastSpy).not.toHaveBeenCalled();
  });

  test("surfaces the error detail when the mutation throws", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() => hoisted.captured.mutation?.onError(new Error("network blip")));
    expect(result.current.result?.message).toContain("network blip");
    expect(hoisted.toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Error", variant: "destructive" }),
    );
  });

  test("puts the server detail in the toast instead of a generic retry hint", () => {
    renderHook(() => useGenerateTestDataButton());
    act(() =>
      hoisted.captured.mutation?.onError(
        new Error(
          "Test data generation is only available in local environments.",
        ),
      ),
    );
    expect(hoisted.toastSpy).toHaveBeenCalledWith(
      expect.objectContaining({
        description:
          "Test data generation is only available in local environments.",
      }),
    );
  });

  test("falls back to a generic message when the error carries no detail", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() => hoisted.captured.mutation?.onError({}));
    expect(result.current.result?.message).toContain(
      "Failed to generate test data. Please try again.",
    );
  });

  test("openDialog clears any prior result and opens the dialog", () => {
    const { result } = renderHook(() => useGenerateTestDataButton());
    act(() =>
      hoisted.captured.mutation?.onSuccess({
        status: 200,
        data: { success: true, message: "done" },
      }),
    );
    act(() => result.current.openDialog());
    expect(result.current.isDialogOpen).toBe(true);
    expect(result.current.result).toBeNull();
  });
});
