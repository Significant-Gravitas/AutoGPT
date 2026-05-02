import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { useChatInput } from "../useChatInput";

vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    initialPrompt: null,
    setInitialPrompt: vi.fn(),
  }),
}));

describe("useChatInput", () => {
  const mockOnSend = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockOnSend.mockResolvedValue(undefined);
  });

  it("does not send when value is empty", async () => {
    const { result } = renderHook(() => useChatInput({ onSend: mockOnSend }));

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockOnSend).not.toHaveBeenCalled();
  });

  it("sends trimmed value and clears input", async () => {
    const { result } = renderHook(() => useChatInput({ onSend: mockOnSend }));

    act(() => {
      result.current.setValue("  hello  ");
    });

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockOnSend).toHaveBeenCalledWith("hello");
    expect(result.current.value).toBe("");
  });

  it("submits the live form value when React state has not caught up", async () => {
    const { result } = renderHook(() => useChatInput({ onSend: mockOnSend }));
    const form = document.createElement("form");
    const textarea = document.createElement("textarea");
    textarea.name = "message";
    textarea.value = "  visible textarea value  ";
    form.append(textarea);

    act(() => {
      result.current.setValue("");
    });

    act(() => {
      result.current.handleSubmit({
        preventDefault: vi.fn(),
        currentTarget: form,
      } as unknown as Parameters<typeof result.current.handleSubmit>[0]);
    });

    await waitFor(() => {
      expect(mockOnSend).toHaveBeenCalledWith("visible textarea value");
    });
  });

  it("does not send when disabled", async () => {
    const { result } = renderHook(() =>
      useChatInput({ onSend: mockOnSend, disabled: true }),
    );

    act(() => {
      result.current.setValue("hello");
    });

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockOnSend).not.toHaveBeenCalled();
  });

  it("prevents double-submit via ref guard", async () => {
    let resolveFirst: () => void;
    const slowSend = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          resolveFirst = resolve;
        }),
    );

    const { result } = renderHook(() => useChatInput({ onSend: slowSend }));

    act(() => {
      result.current.setValue("hello");
    });

    act(() => {
      void result.current.handleSend();
    });

    await act(async () => {
      await result.current.handleSend();
    });

    expect(slowSend).toHaveBeenCalledTimes(1);

    await act(async () => {
      resolveFirst!();
    });
  });

  it("allows sending empty when canSendEmpty is true", async () => {
    const { result } = renderHook(() =>
      useChatInput({ onSend: mockOnSend, canSendEmpty: true }),
    );

    await act(async () => {
      await result.current.handleSend();
    });

    expect(mockOnSend).toHaveBeenCalledWith("");
  });

  it("resets isSending after onSend throws", async () => {
    mockOnSend.mockRejectedValue(new Error("fail"));

    const { result } = renderHook(() => useChatInput({ onSend: mockOnSend }));

    act(() => {
      result.current.setValue("hello");
    });

    await act(async () => {
      try {
        await result.current.handleSend();
      } catch {
        // expected
      }
    });

    expect(result.current.isSending).toBe(false);
  });
});
