import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "./store";
import { useWorkflowImportAutoSubmit } from "./useWorkflowImportAutoSubmit";

const PROMPT = "Build this workflow";

function storeAutoSubmitPrompt() {
  window.sessionStorage.setItem("importWorkflowPrompt", PROMPT);
  window.sessionStorage.setItem("importWorkflowAutosubmit", "true");
}

beforeEach(() => {
  window.sessionStorage.clear();
  window.history.replaceState({}, "", "/copilot");
  useCopilotUIStore.getState().setInitialPrompt(null);
});

describe("useWorkflowImportAutoSubmit", () => {
  it("waits for unresolved expert identity before consuming or submitting the prompt", async () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    const setPendingFileParts = vi.fn();
    storeAutoSubmitPrompt();

    const { rerender } = renderHook(
      ({ isSendLocked }) =>
        useWorkflowImportAutoSubmit({
          onSend,
          setPendingFileParts,
          isSendLocked,
        }),
      { initialProps: { isSendLocked: true } },
    );

    expect(onSend).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem("importWorkflowPrompt")).toBe(PROMPT);

    rerender({ isSendLocked: false });

    await waitFor(() => expect(onSend).toHaveBeenCalledWith(PROMPT));
    expect(window.sessionStorage.getItem("importWorkflowPrompt")).toBeNull();
  });

  it("does not consume or submit an auto-submit prompt for an archived expert", () => {
    const onSend = vi.fn().mockResolvedValue(undefined);
    const setPendingFileParts = vi.fn();
    storeAutoSubmitPrompt();

    renderHook(() =>
      useWorkflowImportAutoSubmit({
        onSend,
        setPendingFileParts,
        isSendLocked: true,
      }),
    );

    expect(onSend).not.toHaveBeenCalled();
    expect(setPendingFileParts).not.toHaveBeenCalled();
    expect(window.sessionStorage.getItem("importWorkflowPrompt")).toBe(PROMPT);
  });

  describe("a prompt from a suggested task", () => {
    function landWithTaskPrompt() {
      window.history.replaceState(
        {},
        "",
        "/home?autosubmit=true&suggestedTaskId=task-1#prompt=Do%20it",
      );
    }

    it("reports the task and its chat once the chat exists", async () => {
      const onTaskStarted = vi.fn();
      landWithTaskPrompt();

      renderHook(() =>
        useWorkflowImportAutoSubmit({
          onSend: vi.fn().mockResolvedValue("session-1"),
          setPendingFileParts: vi.fn(),
          onTaskStarted,
        }),
      );

      await waitFor(() =>
        expect(onTaskStarted).toHaveBeenCalledWith({
          taskID: "task-1",
          sessionID: "session-1",
        }),
      );
      expect(window.location.search).toBe("");
    });

    it.each([
      ["no chat came of it", () => Promise.resolve(undefined)],
      ["the send failed", () => Promise.reject(new Error("boom"))],
    ])("leaves the task open when %s", async (_name, send) => {
      const onSend = vi.fn(send);
      const onTaskStarted = vi.fn();
      landWithTaskPrompt();

      renderHook(() =>
        useWorkflowImportAutoSubmit({
          onSend,
          setPendingFileParts: vi.fn(),
          onTaskStarted,
        }),
      );

      await waitFor(() => expect(onSend).toHaveBeenCalled());
      await new Promise((resolve) => setTimeout(resolve, 20));
      expect(onTaskStarted).not.toHaveBeenCalled();
    });
  });
});
