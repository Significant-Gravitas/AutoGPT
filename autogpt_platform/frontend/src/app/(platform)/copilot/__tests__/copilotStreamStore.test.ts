import type { FileUIPart } from "ai";
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
