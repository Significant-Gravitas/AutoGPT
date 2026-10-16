import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
const auth = vi.hoisted(() => ({
  user: { id: "user-a" } as { id: string } | null,
}));
const prompts = vi.hoisted(() => ({ initialPrompt: null as string | null }));
vi.mock("@/lib/auth/hooks/useAuthStore", () => ({
  useAuthStore: (selector: (state: typeof auth) => unknown) => selector(auth),
}));
vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: () => ({
    initialPrompt: prompts.initialPrompt,
    setInitialPrompt: (value: string | null) => {
      prompts.initialPrompt = value;
    },
    notifyMessageSent: vi.fn(),
  }),
}));
import { useChatInput } from "../useChatInput";

beforeEach(() => {
  sessionStorage.clear();
  auth.user = { id: "user-a" };
  prompts.initialPrompt = null;
});
describe("platform chat drafts", () => {
  it("restores the same account and conversation after a billing navigation remount", () => {
    const first = renderHook(() =>
      useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
    );
    act(() => first.result.current.setValue("My unsent next step"));
    first.unmount();
    const returned = renderHook(() =>
      useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
    );
    expect(returned.result.current.value).toBe("My unsent next step");
  });
  it("isolates users and sessions even without unmounting", () => {
    const { result, rerender } = renderHook(
      ({ draftKey }) => useChatInput({ onSend: vi.fn(), draftKey }),
      { initialProps: { draftKey: "session:chat-a" } },
    );
    act(() => result.current.setValue("Private draft"));
    rerender({ draftKey: "session:chat-b" });
    expect(result.current.value).toBe("");
    auth.user = { id: "user-b" };
    rerender({ draftKey: "session:chat-a" });
    expect(result.current.value).toBe("");
    auth.user = { id: "user-a" };
    rerender({ draftKey: "session:chat-a" });
    expect(result.current.value).toBe("Private draft");
  });
  it("clears a successfully sent draft so returning does not resurrect it", async () => {
    const first = renderHook(() =>
      useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
    );
    act(() => first.result.current.setValue("Send this"));
    await act(() => first.result.current.handleSend());
    first.unmount();
    expect(
      renderHook(() =>
        useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
      ).result.current.value,
    ).toBe("");
  });
  it("persists the restored draft after a failed send", async () => {
    const first = renderHook(() =>
      useChatInput({
        onSend: () => {
          throw new Error("offline");
        },
        draftKey: "session:chat-a",
      }),
    );
    act(() => first.result.current.setValue("Keep this after failure"));
    await act(() => first.result.current.handleSend());
    first.unmount();
    expect(
      renderHook(() =>
        useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
      ).result.current.value,
    ).toBe("Keep this after failure");
  });
  it("merges a late send failure with the draft written after returning", async () => {
    let rejectSend: ((error: Error) => void) | undefined;
    const sending = new Promise<void>((_, reject) => {
      rejectSend = reject;
    });
    const first = renderHook(() =>
      useChatInput({
        onSend: () => sending,
        draftKey: "session:chat-a",
      }),
    );
    act(() => first.result.current.setValue("Original message"));
    let send: Promise<void> | undefined;
    act(() => {
      send = first.result.current.handleSend();
    });
    first.unmount();
    const returned = renderHook(() =>
      useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
    );
    act(() => returned.result.current.setValue("New draft after returning"));
    await act(async () => {
      rejectSend?.(new Error("offline"));
      await send;
    });
    expect(returned.result.current.value).toBe(
      "Original message\n\nNew draft after returning",
    );
    returned.unmount();
    expect(
      renderHook(() =>
        useChatInput({ onSend: vi.fn(), draftKey: "session:chat-a" }),
      ).result.current.value,
    ).toBe("Original message\n\nNew draft after returning");
  });
  it("keeps a late failed send out of another account's current draft", async () => {
    let rejectSend: ((error: Error) => void) | undefined;
    const sending = new Promise<void>((_, reject) => {
      rejectSend = reject;
    });
    const { result, rerender } = renderHook(() =>
      useChatInput({ onSend: () => sending, draftKey: "session:chat-a" }),
    );
    act(() => result.current.setValue("Account A message"));
    let send: Promise<void> | undefined;
    act(() => {
      send = result.current.handleSend();
    });
    auth.user = { id: "user-b" };
    rerender();
    act(() => result.current.setValue("Account B draft"));
    await act(async () => {
      rejectSend?.(new Error("offline"));
      await send;
    });
    expect(result.current.value).toBe("Account B draft");
    auth.user = { id: "user-a" };
    rerender();
    expect(result.current.value).toBe("Account A message");
  });
  it("allows an explicit guided prompt to replace the restored draft", () => {
    const first = renderHook(() =>
      useChatInput({ onSend: vi.fn(), draftKey: "new:default" }),
    );
    act(() => first.result.current.setValue("Older draft"));
    first.unmount();
    prompts.initialPrompt = "Build a scheduled task";
    expect(
      renderHook(() =>
        useChatInput({ onSend: vi.fn(), draftKey: "new:default" }),
      ).result.current.value,
    ).toBe("Build a scheduled task");
  });
  it("does not persist inputs that have not opted into a platform draft key", () => {
    const first = renderHook(() => useChatInput({ onSend: vi.fn() }));
    act(() => first.result.current.setValue("Tour text"));
    first.unmount();
    expect(sessionStorage.length).toBe(0);
    expect(
      renderHook(() => useChatInput({ onSend: vi.fn() })).result.current.value,
    ).toBe("");
  });
});
