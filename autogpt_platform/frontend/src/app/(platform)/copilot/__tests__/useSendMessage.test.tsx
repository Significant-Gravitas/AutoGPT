import { act, renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotStreamStore } from "../copilotStreamStore";
import { useCopilotUIStore } from "../store";
import { useSendMessage } from "../useSendMessage";
import {
  MAX_ATTACHMENTS,
  type WorkspaceAttachment,
} from "../helpers/workspaceAttachments";

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
const OTHER_SESSION_ID = "4f8b0f7e-9f30-4a3b-a6a1-000000000002";
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
  useCopilotUIStore.getState().setInitialPrompt(null);
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
        [
          {
            kind: "workspace" as const,
            fileId: FILE_ID,
            name: "notes.txt",
            mimeType: "text/plain",
          },
        ],
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
        [
          {
            kind: "workspace" as const,
            fileId: FILE_ID,
            name: "notes.txt",
            mimeType: "text/plain",
          },
        ],
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
    const { result } = renderSendMessage(OTHER_SESSION_ID);

    act(() => {
      useCopilotStreamStore.getState().setPendingUploadSend(SESSION_ID, {
        text: "elsewhere",
        attachments: [],
      });
    });

    await waitFor(() => expect(result.current.pendingSend).toBeNull());
    expect(result.current.isUploadingFiles).toBe(false);
  });
});

describe("useSendMessage placeholders across sessions", () => {
  function uploadResult(name: string) {
    return { file_id: FILE_ID, name, mime_type: "application/pdf" };
  }

  it("keeps one placeholder per session and clears only the one that settles", async () => {
    const uploadA = deferred<ReturnType<typeof uploadResult>>();
    const uploadB = deferred<ReturnType<typeof uploadResult>>();
    uploadFileDirectMock.mockImplementation((file: File) =>
      file.name === "a.pdf" ? uploadA.promise : uploadB.promise,
    );
    const chatA = renderSendMessage(SESSION_ID);
    const chatB = renderSendMessage(OTHER_SESSION_ID);

    act(() => {
      void chatA.result.current.onSend("from A", [makeFile("a.pdf")]);
    });
    act(() => {
      void chatB.result.current.onSend("from B", [makeFile("b.pdf")]);
    });

    await waitFor(() =>
      expect(chatA.result.current.pendingSend?.text).toBe("from A"),
    );
    await waitFor(() =>
      expect(chatB.result.current.pendingSend?.text).toBe("from B"),
    );
    expect(chatA.result.current.isUploadingFiles).toBe(true);
    expect(chatB.result.current.isUploadingFiles).toBe(true);

    await act(async () => {
      uploadA.resolve(uploadResult("a.pdf"));
    });

    await waitFor(() => expect(chatA.sendMessage).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(chatA.result.current.pendingSend).toBeNull());
    expect(chatA.result.current.isUploadingFiles).toBe(false);
    expect(chatB.result.current.pendingSend?.text).toBe("from B");
    expect(chatB.result.current.isUploadingFiles).toBe(true);
    expect(chatB.sendMessage).not.toHaveBeenCalled();

    await act(async () => {
      uploadB.resolve(uploadResult("b.pdf"));
    });

    await waitFor(() => expect(chatB.sendMessage).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(chatB.result.current.pendingSend).toBeNull());
    expect(chatB.result.current.isUploadingFiles).toBe(false);
  });

  it("keeps the placeholder and the upload lock across a remount mid-upload", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const first = renderSendMessage();

    act(() => {
      void first.result.current.onSend("hold on", [makeFile("a.pdf")]);
    });
    await waitFor(() =>
      expect(first.result.current.isUploadingFiles).toBe(true),
    );
    first.unmount();

    const second = renderSendMessage();
    expect(second.result.current.pendingSend?.text).toBe("hold on");
    expect(second.result.current.isUploadingFiles).toBe(true);
  });

  it("moves the first send's placeholder onto the session once it is created", async () => {
    const upload = deferred<ReturnType<typeof uploadResult>>();
    uploadFileDirectMock.mockReturnValue(upload.promise);
    const creation = deferred<string>();
    const newChat = renderSendMessage(null);
    newChat.createSession.mockImplementation(async () => {
      const id = await creation.promise;
      useCopilotStreamStore.getState().bindPendingFirstSendToSession(id);
      return id;
    });

    act(() => {
      void newChat.result.current.onSend("first", [makeFile("a.pdf")]);
    });

    await waitFor(() => expect(newChat.createSession).toHaveBeenCalledTimes(1));
    expect(newChat.result.current.pendingSend?.text).toBe("first");
    expect(newChat.result.current.isUploadingFiles).toBe(true);

    // An existing chat open elsewhere must not pick up the unbound send.
    const other = renderSendMessage(OTHER_SESSION_ID);
    expect(other.result.current.pendingSend).toBeNull();
    expect(other.result.current.isUploadingFiles).toBe(false);

    await act(async () => {
      creation.resolve(SESSION_ID);
    });
    newChat.unmount();

    const created = renderSendMessage(SESSION_ID);
    expect(created.result.current.pendingSend?.text).toBe("first");
    expect(created.result.current.isUploadingFiles).toBe(true);
    await waitFor(() => expect(uploadFileDirectMock).toHaveBeenCalledTimes(1));
    expect(other.result.current.pendingSend).toBeNull();

    await act(async () => {
      upload.resolve(uploadResult("a.pdf"));
    });

    await waitFor(() => expect(created.sendMessage).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(created.result.current.pendingSend).toBeNull());
    expect(created.result.current.isUploadingFiles).toBe(false);
    expect(other.result.current.pendingSend).toBeNull();
  });
});

describe("useSendMessage first send failing after the session exists", () => {
  // Creating a session remounts the chat host, so the queued first send is
  // dispatched by a fresh hook long after `onSend` resolved. Nothing is left
  // to catch its rejection except the hook itself.
  async function startFirstSend(
    text: string,
    files: File[],
    workspaceFiles?: WorkspaceAttachment[],
  ) {
    const newChat = renderSendMessage(null);
    newChat.createSession.mockImplementation(async () => {
      useCopilotStreamStore
        .getState()
        .bindPendingFirstSendToSession(SESSION_ID);
      return SESSION_ID;
    });

    await act(async () => {
      await newChat.result.current.onSend(text, files, workspaceFiles);
    });
    newChat.unmount();

    return renderSendMessage(SESSION_ID);
  }

  it("restores the draft when every upload for the first send fails", async () => {
    uploadFileDirectMock.mockRejectedValue(new Error("network down"));

    const created = await startFirstSend("first", [makeFile("talk.pdf")]);

    await waitFor(() =>
      expect(useCopilotUIStore.getState().initialPrompt).toBe("first"),
    );
    expect(created.sendMessage).not.toHaveBeenCalled();
    expect(created.result.current.pendingSend).toBeNull();
    expect(created.result.current.isUploadingFiles).toBe(false);
    expect(toastMock).toHaveBeenLastCalledWith({
      title: "Couldn't send message",
      description:
        "All file uploads failed — your message is back in the composer.",
      variant: "destructive",
    });
  });

  it("puts the workspace references back when the first send's message fails", async () => {
    const created = await startFirstSend(
      "compare",
      [],
      [
        {
          kind: "workspace" as const,
          fileId: FILE_ID,
          name: "notes.txt",
          mimeType: "text/plain",
        },
      ],
    );

    await waitFor(() => expect(created.sendMessage).toHaveBeenCalledTimes(1));
    expect(useCopilotStreamStore.getState().pendingFileParts).toEqual([]);

    await act(async () => {
      created.stream.reject(new Error("stream died"));
    });

    await waitFor(() =>
      expect(useCopilotStreamStore.getState().pendingFileParts).toEqual([
        {
          type: "file",
          mediaType: "text/plain",
          filename: "notes.txt",
          url: `/api/proxy/api/workspace/files/${FILE_ID}/download`,
        },
      ]),
    );
    expect(useCopilotUIStore.getState().initialPrompt).toBe("compare");
  });
});

describe("useSendMessage when creating the first chat's session fails", () => {
  it("clears the placeholder so a failed first send leaves nothing behind", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result, createSession } = renderSendMessage(null);
    createSession.mockRejectedValue(new Error("no session"));

    let failure: unknown;
    await act(async () => {
      failure = await result.current
        .onSend("first", [makeFile("talk.pdf")])
        .catch((err: unknown) => err);
    });

    expect(failure).toBeInstanceOf(Error);
    expect(result.current.pendingSend).toBeNull();
    expect(result.current.isUploadingFiles).toBe(false);
    expect(useCopilotStreamStore.getState().pendingUploadSends).toEqual({});
  });

  it("clears a placeholder already bound to the new session", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result, createSession } = renderSendMessage(null);
    createSession.mockImplementation(async () => {
      // Binding moves the placeholder off the unbound key; failing after it
      // must not leave the session-keyed one stranded.
      useCopilotStreamStore
        .getState()
        .bindPendingFirstSendToSession(SESSION_ID);
      throw new Error("no session");
    });

    await act(async () => {
      await result.current
        .onSend("first", [makeFile("talk.pdf")])
        .catch(() => undefined);
    });

    expect(useCopilotStreamStore.getState().pendingUploadSends).toEqual({});
    expect(renderSendMessage(SESSION_ID).result.current.pendingSend).toBeNull();
  });
});

function makeWorkspaceRefs(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    kind: "workspace" as const,
    fileId: `${FILE_ID.slice(0, -1)}${i}`,
    name: `ws-${i}.txt`,
    mimeType: "text/plain",
  }));
}

describe("useSendMessage send-time cap backstop", () => {
  // The composer refuses the extra file as it is added, so this path is
  // unreachable through the UI — it stays for any caller that is not the
  // composer, and for a state forced past the cap.
  it("refuses a batch over the cap without uploading or sending", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result, sendMessage } = renderSendMessage();

    await act(async () => {
      await result.current.onSend(
        "too much",
        Array.from({ length: MAX_ATTACHMENTS + 1 }, (_, i) =>
          makeFile(`over-${i}.pdf`),
        ),
      );
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Too many attachments" }),
    );
    expect(uploadFileDirectMock).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("refuses a workspace-only batch over the cap", async () => {
    const { result, sendMessage } = renderSendMessage();

    await act(async () => {
      await result.current.onSend(
        "workspace only",
        undefined,
        makeWorkspaceRefs(MAX_ATTACHMENTS + 1),
      );
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Too many attachments" }),
    );
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("counts local and workspace attachments together", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result, sendMessage } = renderSendMessage();

    await act(async () => {
      await result.current.onSend(
        "mixed",
        // Under the cap on its own, over it once the references are counted.
        Array.from({ length: 6 }, (_, i) => makeFile(`local-${i}.pdf`)),
        makeWorkspaceRefs(MAX_ATTACHMENTS - 5),
      );
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({ title: "Too many attachments" }),
    );
    expect(uploadFileDirectMock).not.toHaveBeenCalled();
    expect(sendMessage).not.toHaveBeenCalled();
  });

  it("lets a mixed batch that exactly fills the cap through", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result } = renderSendMessage();

    act(() => {
      void result.current.onSend(
        "mixed, but fits",
        Array.from({ length: 6 }, (_, i) => makeFile(`local-${i}.pdf`)),
        makeWorkspaceRefs(MAX_ATTACHMENTS - 6),
      );
    });

    await waitFor(() => expect(uploadFileDirectMock).toHaveBeenCalledTimes(6));
    expect(toastMock).not.toHaveBeenCalled();
  });

  it("lets exactly the cap through", async () => {
    uploadFileDirectMock.mockReturnValue(new Promise(() => undefined));
    const { result } = renderSendMessage();

    act(() => {
      void result.current.onSend(
        "just enough",
        Array.from({ length: MAX_ATTACHMENTS }, (_, i) =>
          makeFile(`ok-${i}.pdf`),
        ),
      );
    });

    await waitFor(() =>
      expect(uploadFileDirectMock).toHaveBeenCalledTimes(MAX_ATTACHMENTS),
    );
    expect(toastMock).not.toHaveBeenCalled();
  });
});
