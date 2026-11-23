import { useCaptureMarketingPrompt } from "@/hooks/useCaptureMarketingPrompt";
import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "./store";
import { useWorkflowImportAutoSubmit } from "./useWorkflowImportAutoSubmit";

const PROMPT = "Build this workflow";
const SESSION_ID = "7d1c5a0e-3f2b-4c6d-9e8f-0a1b2c3d4e5f";

function storeAutoSubmitPrompt() {
  window.sessionStorage.setItem("importWorkflowPrompt", PROMPT);
  window.sessionStorage.setItem("importWorkflowAutosubmit", "true");
}

function renderOnOpenChat() {
  const onSend = vi.fn().mockResolvedValue(undefined);
  const setPendingFileParts = vi.fn();
  renderHook(() =>
    useWorkflowImportAutoSubmit({ onSend, setPendingFileParts }),
  );
  return { onSend, setPendingFileParts };
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

  it("prefills a prompt that arrives in the URL rather than sending it", () => {
    window.history.replaceState(
      {},
      "",
      `/home?sessionId=${SESSION_ID}&autosubmit=true#prompt=${encodeURIComponent(PROMPT)}`,
    );

    const { onSend } = renderOnOpenChat();

    expect(onSend).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().initialPrompt).toBe(PROMPT);
  });

  it("prefills a stashed prompt the app did not ask to send, whatever the URL says", () => {
    window.history.replaceState(
      {},
      "",
      `/home?sessionId=${SESSION_ID}&autosubmit=true`,
    );
    window.sessionStorage.setItem("importWorkflowPrompt", PROMPT);

    const { onSend } = renderOnOpenChat();

    expect(onSend).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().initialPrompt).toBe(PROMPT);
  });

  it("prefills the marketing hand-off its own capture hook stashed", () => {
    window.history.replaceState(
      {},
      "",
      `/login?triggerprompt=${encodeURIComponent(PROMPT)}`,
    );
    renderHook(() => useCaptureMarketingPrompt());
    expect(window.sessionStorage.getItem("importWorkflowPrompt")).toBe(PROMPT);

    window.history.replaceState({}, "", `/home?sessionId=${SESSION_ID}`);
    const { onSend } = renderOnOpenChat();

    expect(onSend).not.toHaveBeenCalled();
    expect(useCopilotUIStore.getState().initialPrompt).toBe(PROMPT);
  });
});
