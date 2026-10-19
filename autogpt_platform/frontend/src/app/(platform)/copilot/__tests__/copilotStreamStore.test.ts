import type { FileUIPart, UIMessage } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const STORAGE_KEY = "copilot-stream-store";

async function importFreshStore() {
  vi.resetModules();
  return import("../copilotStreamStore");
}

describe("copilotStreamStore pending first send persistence", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  afterEach(() => {
    vi.resetModules();
    window.sessionStorage.clear();
  });

  it("restores a bound text send and workspace parts for its target exactly once", async () => {
    const firstModule = await importFreshStore();
    const workspacePart: FileUIPart = {
      type: "file",
      mediaType: "text/plain",
      filename: "workspace-note.txt",
      url: "/api/workspace/files/workspace-file-id/download",
    };
    const firstStore = firstModule.useCopilotStreamStore.getState();

    firstStore.setPendingFirstSend({
      text: "hello from mobile",
      files: [],
    });
    firstStore.setPendingFileParts([workspacePart]);
    firstStore.bindPendingFirstSendToSession("target-session");

    const secondModule = await importFreshStore();
    expect(
      secondModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("target-session"),
    ).toEqual({
      send: { text: "hello from mobile", files: [] },
      parts: [workspacePart],
    });
    expect(
      secondModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("target-session"),
    ).toEqual({ send: null, parts: [] });

    const thirdModule = await importFreshStore();
    expect(
      thirdModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("target-session"),
    ).toEqual({ send: null, parts: [] });
  });

  it("restores expert kickoff metadata across the session-creation remount", async () => {
    const firstModule = await importFreshStore();
    const metadata = {
      kind: "expert_kickoff" as const,
      expertId: "3f8b0f7e-9f30-4a3b-a6a1-000000000001",
      attemptToken: "attempt-1",
    };
    const firstStore = firstModule.useCopilotStreamStore.getState();
    firstStore.setPendingFirstSend({
      text: "private kickoff prompt",
      files: [],
      metadata,
    });
    firstStore.bindPendingFirstSendToSession("expert-session");

    const secondModule = await importFreshStore();
    expect(
      secondModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("expert-session"),
    ).toEqual({
      send: { text: "private kickoff prompt", files: [], metadata },
      parts: [],
    });
  });

  it("does not consume a pending send for a different session", async () => {
    const firstModule = await importFreshStore();
    const firstStore = firstModule.useCopilotStreamStore.getState();
    firstStore.setPendingFirstSend({ text: "targeted text", files: [] });
    firstStore.bindPendingFirstSendToSession("intended-session");

    const secondModule = await importFreshStore();
    expect(
      secondModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("other-session"),
    ).toEqual({ send: null, parts: [] });

    const thirdModule = await importFreshStore();
    expect(
      thirdModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("intended-session"),
    ).toEqual({
      send: { text: "targeted text", files: [] },
      parts: [],
    });
  });

  it("keeps local File pending state in memory and omits it from storage", async () => {
    const { useCopilotStreamStore } = await importFreshStore();
    const localFile = new File(["private local contents"], "local-secret.txt", {
      type: "text/plain",
    });
    const store = useCopilotStreamStore.getState();

    store.setPendingFirstSend({
      text: "send this text and file",
      files: [localFile],
    });
    store.setPendingFileParts([
      {
        type: "file",
        mediaType: "text/plain",
        filename: "workspace-note.txt",
        url: "/api/workspace/files/workspace-file-id/download",
      },
    ]);
    store.bindPendingFirstSendToSession("target-session");

    expect(useCopilotStreamStore.getState().pendingFirstSend?.files).toEqual([
      localFile,
    ]);
    const stored = window.sessionStorage.getItem(STORAGE_KEY);
    expect(stored).not.toBeNull();
    expect(stored).not.toContain("local-secret.txt");
    expect(stored).not.toContain("private local contents");
    expect(stored).not.toContain("workspace-note.txt");
    expect(JSON.parse(stored ?? "{}").state).toMatchObject({
      pendingFirstSend: null,
      pendingFirstSendSessionId: null,
      pendingFileParts: [],
    });

    const freshModule = await importFreshStore();
    expect(
      freshModule.useCopilotStreamStore
        .getState()
        .takePendingFirstSend("target-session"),
    ).toEqual({ send: null, parts: [] });
  });

  it("binding without pending input is a no-op", async () => {
    const { useCopilotStreamStore } = await importFreshStore();
    useCopilotStreamStore
      .getState()
      .bindPendingFirstSendToSession("manual-session");

    expect(
      useCopilotStreamStore.getState().pendingFirstSendSessionId,
    ).toBeNull();
    expect(
      useCopilotStreamStore.getState().takePendingFirstSend("manual-session"),
    ).toEqual({ send: null, parts: [] });
  });
});

const SESSION = "snapshot-session";

function userMessage(): UIMessage {
  return { id: "u1", role: "user", parts: [{ type: "text", text: "go" }] };
}

function assistantMessage(parts: UIMessage["parts"]): UIMessage {
  return { id: "a1", role: "assistant", parts };
}

function textOnly(text: string): UIMessage[] {
  return [userMessage(), assistantMessage([{ type: "text", text }])];
}

describe("copilotStreamStore message snapshots", () => {
  beforeEach(() => {
    window.sessionStorage.clear();
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.resetModules();
    window.sessionStorage.clear();
  });

  async function freshStoreWithSnapshot(messages: UIMessage[]) {
    const { useCopilotStreamStore } = await importFreshStore();
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, messages);
    const listener = vi.fn();
    useCopilotStreamStore.subscribe(listener);
    const setItem = vi.spyOn(Storage.prototype, "setItem");
    return { useCopilotStreamStore, listener, setItem };
  }

  it("does not publish a streamed token that only adds text", async () => {
    const first = textOnly("Hel");
    const { useCopilotStreamStore, listener, setItem } =
      await freshStoreWithSnapshot(first);

    const next = textOnly("Hello");
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, next);
    const withNewMessage = [
      ...next,
      { id: "a2", role: "assistant", parts: [{ type: "text", text: "More" }] },
    ] satisfies UIMessage[];
    useCopilotStreamStore
      .getState()
      .setMessageSnapshot(SESSION, withNewMessage);

    expect(listener).not.toHaveBeenCalled();
    expect(setItem).not.toHaveBeenCalled();
    expect(useCopilotStreamStore.getState().messageSnapshots[SESSION]).toBe(
      first,
    );
    expect(useCopilotStreamStore.getState().getMessageSnapshot(SESSION)).toBe(
      withNewMessage,
    );
  });

  it("ignores a repeated write of the published messages", async () => {
    const first = textOnly("Hel");
    const { useCopilotStreamStore, listener } =
      await freshStoreWithSnapshot(first);
    const store = useCopilotStreamStore.getState();

    store.setMessageSnapshot(SESSION, textOnly("Hello"));
    store.setMessageSnapshot(SESSION, first);

    expect(listener).not.toHaveBeenCalled();
    expect(store.getMessageSnapshot(SESSION)).toBe(first);
  });

  it("publishes a message that gains a file part", async () => {
    const { useCopilotStreamStore, listener } = await freshStoreWithSnapshot(
      textOnly("Here is the file"),
    );

    const next = [
      userMessage(),
      assistantMessage([
        { type: "text", text: "Here is the file" },
        {
          type: "file",
          mediaType: "text/markdown",
          filename: "brief.md",
          url: "/api/proxy/api/workspace/files/bbbbbbbb-0000-0000-0000-000000000002/download",
        },
      ]),
    ];
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, next);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(useCopilotStreamStore.getState().messageSnapshots[SESSION]).toBe(
      next,
    );
  });

  it("publishes streamed text that links a workspace file", async () => {
    const { useCopilotStreamStore, listener } = await freshStoreWithSnapshot(
      textOnly("Here is [the brief]("),
    );

    const next = textOnly(
      "Here is [the brief](workspace://aaaaaaaa-0000-0000-0000-000000000001)",
    );
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, next);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(useCopilotStreamStore.getState().messageSnapshots[SESSION]).toBe(
      next,
    );
  });

  it("publishes a tool call that starts a run", async () => {
    const { useCopilotStreamStore, listener } = await freshStoreWithSnapshot(
      textOnly("Starting it"),
    );

    const next = [
      userMessage(),
      assistantMessage([
        { type: "text", text: "Starting it" },
        {
          type: "tool-run_agent",
          toolCallId: "call-1",
          state: "output-available",
          input: {},
          output: { execution_id: "exec-1", status: "QUEUED" },
        },
      ]),
    ];
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, next);

    expect(listener).toHaveBeenCalledTimes(1);
    expect(useCopilotStreamStore.getState().messageSnapshots[SESSION]).toBe(
      next,
    );
  });

  it("publishes every write of messages it cannot digest", async () => {
    const { useCopilotStreamStore, listener } = await freshStoreWithSnapshot(
      textOnly("Hel"),
    );
    const malformed = {
      type: "text",
    } as unknown as UIMessage["parts"][number];

    const first = [userMessage(), assistantMessage([malformed])];
    const second = [userMessage(), assistantMessage([malformed])];
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, first);
    useCopilotStreamStore.getState().setMessageSnapshot(SESSION, second);

    expect(listener).toHaveBeenCalledTimes(2);
    expect(useCopilotStreamStore.getState().messageSnapshots[SESSION]).toBe(
      second,
    );
  });

  it("forgets the unpublished latest snapshot when the session is cleared", async () => {
    const { useCopilotStreamStore } = await freshStoreWithSnapshot(
      textOnly("Hel"),
    );
    const store = useCopilotStreamStore.getState();

    store.setMessageSnapshot(SESSION, textOnly("Hello"));
    store.clearSession(SESSION);
    expect(store.getMessageSnapshot(SESSION)).toEqual([]);

    store.setMessageSnapshot(SESSION, textOnly("Hel"));
    store.setMessageSnapshot(SESSION, textOnly("Hello"));
    store.resetAll();
    expect(store.getMessageSnapshot(SESSION)).toEqual([]);

    store.setMessageSnapshot(SESSION, textOnly("Hel"));
    store.setMessageSnapshot(SESSION, textOnly("Hello"));
    useCopilotStreamStore.setState({ messageSnapshots: {} });
    expect(store.getMessageSnapshot(SESSION)).toEqual([]);
  });
});
