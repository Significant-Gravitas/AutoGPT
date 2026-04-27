import { renderHook } from "@testing-library/react";
import { useRef } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

const mockUploadFileDirect = vi.fn();
vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: (...args: unknown[]) => mockUploadFileDirect(...args),
}));

import { useCopilotStreamStore } from "../copilotStreamStore";
import { useSendMessage } from "../useSendMessage";

function file(name: string, size = 100, type = "text/plain"): File {
  const blob = new Blob([new Uint8Array(size)], { type });
  return new File([blob], name, { type });
}

function setupHook(opts: {
  sessionId: string | null;
  createSession?: () => Promise<string | undefined>;
}) {
  const sendMessage = vi.fn();
  const createSession =
    opts.createSession ??
    vi.fn(async () => "new-session" as string | undefined);

  const { result, rerender } = renderHook(
    ({ sessionId }: { sessionId: string | null }) => {
      const isUserStoppingRef = useRef(true);
      const hook = useSendMessage({
        sessionId,
        sendMessage,
        createSession,
        isUserStoppingRef,
      });
      return { ...hook, isUserStoppingRef };
    },
    { initialProps: { sessionId: opts.sessionId } },
  );
  return { result, rerender, sendMessage, createSession };
}

describe("useSendMessage", () => {
  beforeEach(() => {
    mockToast.mockClear();
    mockUploadFileDirect.mockReset();
    // Reset shared store between tests via the store's own resetAll helper
    // to avoid clobbering its internal action functions.
    useCopilotStreamStore.getState().resetAll();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("does nothing when both text and files are empty", async () => {
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    await result.current.onSend("   ");
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("dispatches a plain text message immediately when a session exists", async () => {
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    await result.current.onSend("hello");
    expect(sendMessage).toHaveBeenCalledWith({ text: "hello" });
  });

  it("flips isUserStoppingRef back to false on each send", async () => {
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    expect(result.current.isUserStoppingRef.current).toBe(true);
    await result.current.onSend("hello");
    expect(result.current.isUserStoppingRef.current).toBe(false);
    expect(sendMessage).toHaveBeenCalledTimes(1);
  });

  it("rejects more than 10 attached files with a destructive toast", async () => {
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    const tooMany = Array.from({ length: 11 }, (_, i) => file(`f${i}.txt`));
    await result.current.onSend("text", tooMany);
    expect(sendMessage).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Too many files",
        variant: "destructive",
      }),
    );
  });

  it("rejects a single file larger than the 100 MB limit", async () => {
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    const big = file("huge.bin", 101 * 1024 * 1024, "application/octet-stream");
    await result.current.onSend("text", [big]);
    expect(sendMessage).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "File too large",
        variant: "destructive",
      }),
    );
  });

  it("uploads files and sends a message with file parts", async () => {
    mockUploadFileDirect.mockResolvedValueOnce({
      file_id: "fid-1",
      name: "doc.txt",
      mime_type: "text/plain",
    });
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    await result.current.onSend("with file", [file("doc.txt")]);

    expect(mockUploadFileDirect).toHaveBeenCalledTimes(1);
    expect(sendMessage).toHaveBeenCalledTimes(1);
    const arg = sendMessage.mock.calls[0][0] as {
      text: string;
      files: {
        type: string;
        mediaType: string;
        filename: string;
        url: string;
      }[];
    };
    expect(arg.text).toBe("with file");
    expect(arg.files).toHaveLength(1);
    expect(arg.files[0]).toMatchObject({
      type: "file",
      filename: "doc.txt",
      mediaType: "text/plain",
    });
    expect(arg.files[0].url).toContain("fid-1");
  });

  it("toasts and throws when every upload fails", async () => {
    mockUploadFileDirect.mockRejectedValue(new Error("network down"));
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    await expect(
      result.current.onSend("with file", [file("a.txt"), file("b.txt")]),
    ).rejects.toThrow();
    expect(sendMessage).not.toHaveBeenCalled();
    // One per-file toast PLUS one aggregate failure toast.
    const titles = mockToast.mock.calls.map(
      (c) => (c[0] as { title: string }).title,
    );
    expect(titles).toContain("File upload failed");
  });

  it("uses pre-built file parts from the store and skips upload", async () => {
    // Mount BEFORE seeding parts — the post-session-creation flush effect
    // calls takePendingFirstSend on mount with a sessionId, which clears
    // pendingFileParts as a side effect.
    const { result, sendMessage } = setupHook({ sessionId: "s1" });
    useCopilotStreamStore.getState().setPendingFileParts([
      {
        type: "file" as const,
        mediaType: "image/png",
        filename: "img.png",
        url: "/api/proxy/x",
      },
    ]);
    await result.current.onSend("see attached");
    expect(mockUploadFileDirect).not.toHaveBeenCalled();
    expect(sendMessage).toHaveBeenCalledWith({
      text: "see attached",
      files: [expect.objectContaining({ type: "file", filename: "img.png" })],
    });
    // Pre-built parts are consumed once.
    expect(useCopilotStreamStore.getState().pendingFileParts).toEqual([]);
  });

  it("stashes a first-send to the store and creates a session when none exists", async () => {
    const createSession = vi.fn(async () => "new-id" as string | undefined);
    const { result } = setupHook({ sessionId: null, createSession });
    await result.current.onSend("hi", [file("a.txt")]);
    expect(createSession).toHaveBeenCalledTimes(1);
    expect(useCopilotStreamStore.getState().pendingFirstSend?.text).toBe("hi");
    expect(
      useCopilotStreamStore.getState().pendingFirstSend?.files,
    ).toHaveLength(1);
  });

  it("blocks a second send while a no-session createSession is in flight", async () => {
    let resolveCreate: (value: string) => void = () => {};
    const createSession = vi.fn(
      () => new Promise<string>((r) => (resolveCreate = r)),
    );
    const { result } = setupHook({ sessionId: null, createSession });

    // Two rapid sends with no session — only the first should make it through.
    const first = result.current.onSend("first");
    const second = result.current.onSend("second");

    // Resolve the in-flight create.
    resolveCreate("new-id");
    await Promise.all([first, second]);

    expect(createSession).toHaveBeenCalledTimes(1);
    // pendingFirstSend reflects the FIRST message — the second short-circuits
    // without overwriting.
    expect(useCopilotStreamStore.getState().pendingFirstSend?.text).toBe(
      "first",
    );
  });

  it("clears the pending first-send when createSession rejects", async () => {
    const createSession = vi.fn(async () => {
      throw new Error("create failed");
    });
    const { result } = setupHook({ sessionId: null, createSession });
    await expect(result.current.onSend("hi")).rejects.toThrow();
    expect(useCopilotStreamStore.getState().pendingFirstSend).toBeNull();
    expect(useCopilotStreamStore.getState().pendingFileParts).toEqual([]);
  });

  it("dispatches the queued first-send once a sessionId becomes available", async () => {
    useCopilotStreamStore.getState().setPendingFirstSend({
      text: "queued",
      files: [],
    });
    const { rerender, sendMessage } = setupHook({ sessionId: null });
    rerender({ sessionId: "now-active" });
    expect(sendMessage).toHaveBeenCalledWith({ text: "queued" });
  });

  it("setPendingFileParts forwards to the store", () => {
    const { result } = setupHook({ sessionId: "s1" });
    result.current.setPendingFileParts([
      {
        type: "file" as const,
        mediaType: "image/jpeg",
        filename: "x.jpg",
        url: "/api/y",
      },
    ]);
    expect(useCopilotStreamStore.getState().pendingFileParts).toHaveLength(1);
  });
});
