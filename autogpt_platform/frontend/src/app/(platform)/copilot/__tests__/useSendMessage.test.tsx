import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useSendMessage } from "../useSendMessage";

const { uploadFileDirectMock, toastMock } = vi.hoisted(() => ({
  uploadFileDirectMock: vi.fn(),
  toastMock: vi.fn(),
}));

vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: uploadFileDirectMock,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: toastMock,
}));

const SESSION_ID = "4f8b0f7e-9f30-4a3b-a6a1-000000000001";
const FILE_ID = "5f8b0f7e-9f30-4a3b-a6a1-000000000001";

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}

function makeFile(name: string, type = "application/pdf") {
  return new File(["x".repeat(2048)], name, { type });
}

function renderSendMessage(sessionId: string | null = SESSION_ID) {
  const stream = deferred<void>();
  const sendMessage = vi.fn(() => stream.promise);
  const createSession = vi.fn(async () => SESSION_ID);
  const hook = renderHook(() =>
    useSendMessage({
      sessionId,
      sendMessage: sendMessage as never,
      createSession,
      isUserStoppingRef: { current: false },
    }),
  );
  return { ...hook, sendMessage, createSession, stream };
}

beforeEach(() => {
  uploadFileDirectMock.mockReset();
  toastMock.mockReset();
  useCopilotStreamStore.getState().resetAll();
});

describe("useSendMessage with local attachments", () => {
  it("publishes a placeholder for the bubble before the upload finishes", async () => {
    const upload = deferred<{
      file_id: string;
      name: string;
      mime_type: string;
    }>();
    uploadFileDirectMock.mockReturnValue(upload.promise);
    const { result, sendMessage } = renderSendMessage();

    let sendPromise: Promise<void> | undefined;
    act(() => {
      sendPromise = result.current.onSend("tell me about this", [
        makeFile("talk.pdf"),
      ]);
    });

    await waitFor(() => expect(result.current.pendingSend).not.toBeNull());
    expect(result.current.pendingSend).toEqual({
      sessionId: SESSION_ID,
      text: "tell me about this",
      attachments: [
        {
          name: "talk.pdf",
          mediaType: "application/pdf",
          sizeBytes: 2048,
          isUploading: true,
        },
      ],
    });
    expect(result.current.isUploadingFiles).toBe(true);
    expect(sendMessage).not.toHaveBeenCalled();

    await act(async () => {
      upload.resolve({
        file_id: FILE_ID,
        name: "talk.pdf",
        mime_type: "application/pdf",
      });
    });

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    expect(sendMessage).toHaveBeenCalledWith({
      text: "tell me about this",
      files: [
        {
          type: "file",
          mediaType: "application/pdf",
          filename: "talk.pdf",
          url: `/api/proxy/api/workspace/files/${FILE_ID}/download`,
        },
      ],
      metadata: undefined,
    });
    // The placeholder and the upload lock both clear once the real message
    // is handed to the SDK — not when the whole stream ends.
    await waitFor(() => expect(result.current.pendingSend).toBeNull());
    expect(result.current.isUploadingFiles).toBe(false);
    expect(sendPromise).toBeDefined();
  });

  it("lists workspace references as already stored next to the uploading files", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result } = renderSendMessage();

    act(() => {
      void result.current.onSend(
        "compare",
        [makeFile("icon.png", "image/png")],
        [{ fileId: FILE_ID, name: "notes.txt", mimeType: "text/plain" }],
      );
    });

    await waitFor(() => expect(result.current.pendingSend).not.toBeNull());
    expect(result.current.pendingSend?.attachments).toEqual([
      { name: "notes.txt", mediaType: "text/plain", isUploading: false },
      {
        name: "icon.png",
        mediaType: "image/png",
        sizeBytes: 2048,
        isUploading: true,
      },
    ]);
  });

  it("drops the placeholder and rethrows when every upload fails", async () => {
    uploadFileDirectMock.mockRejectedValue(new Error("boom"));
    const { result, sendMessage } = renderSendMessage();

    let failure: unknown;
    await act(async () => {
      failure = await result.current
        .onSend("hi", [makeFile("talk.pdf")])
        .catch((err: unknown) => err);
    });

    expect(failure).toBeInstanceOf(Error);
    expect(sendMessage).not.toHaveBeenCalled();
    expect(result.current.pendingSend).toBeNull();
    expect(result.current.isUploadingFiles).toBe(false);
  });

  it("still sends the workspace references when every local upload fails", async () => {
    uploadFileDirectMock.mockRejectedValue(new Error("boom"));
    const { result, sendMessage } = renderSendMessage();

    act(() => {
      void result.current.onSend(
        "compare",
        [makeFile("talk.pdf")],
        [{ fileId: FILE_ID, name: "notes.txt", mimeType: "text/plain" }],
      );
    });

    await waitFor(() => expect(sendMessage).toHaveBeenCalledTimes(1));
    expect(sendMessage).toHaveBeenCalledWith({
      text: "compare",
      files: [
        {
          type: "file",
          mediaType: "text/plain",
          filename: "notes.txt",
          url: `/api/proxy/api/workspace/files/${FILE_ID}/download`,
        },
      ],
      metadata: undefined,
    });
    await waitFor(() => expect(result.current.pendingSend).toBeNull());
    expect(result.current.isUploadingFiles).toBe(false);
  });

  it("does not publish a placeholder for a text-only first message", async () => {
    const creation = deferred<string>();
    const { result, createSession } = renderSendMessage(null);
    createSession.mockReturnValue(creation.promise);

    act(() => {
      void result.current.onSend("just text");
    });

    await waitFor(() => expect(createSession).toHaveBeenCalledTimes(1));
    expect(result.current.pendingSend).toBeNull();
    expect(useCopilotStreamStore.getState().pendingFirstSend?.text).toBe(
      "just text",
    );
  });

  it("shows the placeholder while the first chat's session is still being created", async () => {
    const creation = deferred<string>();
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result, createSession } = renderSendMessage(null);
    createSession.mockReturnValue(creation.promise);

    act(() => {
      void result.current.onSend("first", [makeFile("talk.pdf")]);
    });

    await waitFor(() => expect(createSession).toHaveBeenCalledTimes(1));
    expect(result.current.pendingSend).toEqual({
      sessionId: null,
      text: "first",
      attachments: [
        {
          name: "talk.pdf",
          mediaType: "application/pdf",
          sizeBytes: 2048,
          isUploading: true,
        },
      ],
    });
  });

  it("hides a placeholder that belongs to another session", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result } = renderSendMessage("some-other-session");

    act(() => {
      useCopilotStreamStore.getState().setPendingUploadSend({
        sessionId: SESSION_ID,
        text: "elsewhere",
        attachments: [],
      });
    });

    await waitFor(() => expect(result.current.pendingSend).toBeNull());
  });
});
