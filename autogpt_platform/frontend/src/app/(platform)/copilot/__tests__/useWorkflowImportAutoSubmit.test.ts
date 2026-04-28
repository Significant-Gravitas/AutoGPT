import { cleanup, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockSetInitialPrompt = vi.fn();
vi.mock("../store", () => ({
  useCopilotUIStore: () => ({ setInitialPrompt: mockSetInitialPrompt }),
}));

import { useWorkflowImportAutoSubmit } from "../useWorkflowImportAutoSubmit";

function setLocation(href: string) {
  // happy-dom allows reassigning window.location.href via History API; we
  // can also mutate via URL + replaceState for hash/query coverage.
  const next = new URL(href);
  window.history.replaceState(
    null,
    "",
    `${next.pathname}${next.search}${next.hash}`,
  );
}

describe("useWorkflowImportAutoSubmit", () => {
  beforeEach(() => {
    mockSetInitialPrompt.mockClear();
    sessionStorage.clear();
    setLocation("http://localhost/copilot");
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("does nothing when there is no hash and no sessionStorage prompt", () => {
    const onSend = vi.fn();
    renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts: vi.fn() }),
    );
    expect(onSend).not.toHaveBeenCalled();
    expect(mockSetInitialPrompt).not.toHaveBeenCalled();
  });

  it("populates the initial prompt from a hash without autosubmit", () => {
    setLocation("http://localhost/copilot#prompt=Hello%20world");
    const onSend = vi.fn();
    renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts: vi.fn() }),
    );
    expect(onSend).not.toHaveBeenCalled();
    expect(mockSetInitialPrompt).toHaveBeenCalledWith("Hello world");
    // Hash is cleared after consumption.
    expect(window.location.hash).toBe("");
  });

  it("auto-submits when autosubmit=true is set on the query string", async () => {
    setLocation(
      "http://localhost/copilot?autosubmit=true#prompt=auto%20submit%20me",
    );
    const onSend = vi.fn(async () => {});
    renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts: vi.fn() }),
    );
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("auto submit me"));
    expect(mockSetInitialPrompt).not.toHaveBeenCalled();
    expect(window.location.search).toBe(""); // autosubmit param cleared
  });

  it("falls back to setInitialPrompt when an autosubmit send rejects", async () => {
    setLocation(
      "http://localhost/copilot?autosubmit=true#prompt=auto%20submit%20me",
    );
    const onSend = vi.fn(async () => {
      throw new Error("network down");
    });
    renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts: vi.fn() }),
    );
    await waitFor(() =>
      expect(mockSetInitialPrompt).toHaveBeenCalledWith("auto submit me"),
    );
  });

  it("uses sessionStorage prompt over the hash, including pre-uploaded file part", async () => {
    setLocation("http://localhost/copilot?autosubmit=true#prompt=ignored");
    sessionStorage.setItem("importWorkflowPrompt", "from-storage");
    sessionStorage.setItem(
      "importWorkflowFile",
      JSON.stringify({
        fileId: "11111111-2222-3333-4444-555555555555",
        fileName: "wf.json",
        mimeType: "application/json",
      }),
    );

    const setPendingFileParts = vi.fn();
    const onSend = vi.fn(async () => {});
    renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts }),
    );
    await waitFor(() => expect(onSend).toHaveBeenCalledWith("from-storage"));
    expect(setPendingFileParts).toHaveBeenCalledWith([
      expect.objectContaining({
        type: "file",
        mediaType: "application/json",
        filename: "wf.json",
      }),
    ]);
    expect(sessionStorage.getItem("importWorkflowPrompt")).toBeNull();
    expect(sessionStorage.getItem("importWorkflowFile")).toBeNull();
  });

  it("rejects malformed file ids (not a UUID) and skips the file part", async () => {
    sessionStorage.setItem("importWorkflowPrompt", "with-bad-file");
    sessionStorage.setItem(
      "importWorkflowFile",
      JSON.stringify({
        fileId: "../etc/passwd",
        fileName: "evil",
        mimeType: "application/json",
      }),
    );
    setLocation("http://localhost/copilot?autosubmit=true");
    const setPendingFileParts = vi.fn();
    const onSend = vi.fn(async () => {});
    renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts }),
    );
    await waitFor(() => expect(onSend).toHaveBeenCalled());
    expect(setPendingFileParts).not.toHaveBeenCalled();
  });

  it("skips processing on subsequent renders (only runs once)", () => {
    setLocation("http://localhost/copilot#prompt=run-once");
    const onSend = vi.fn();
    const { rerender } = renderHook(() =>
      useWorkflowImportAutoSubmit({ onSend, setPendingFileParts: vi.fn() }),
    );
    expect(mockSetInitialPrompt).toHaveBeenCalledTimes(1);
    rerender();
    rerender();
    expect(mockSetInitialPrompt).toHaveBeenCalledTimes(1);
  });
});
