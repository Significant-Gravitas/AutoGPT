import { beforeEach, describe, expect, it, vi } from "vitest";

const STORE_KEY = "copilot-scheduled-send-handoff";

async function freshStore() {
  vi.resetModules();
  const { useScheduledSendHandoff } = await import("../scheduledSendHandoff");
  return useScheduledSendHandoff;
}

beforeEach(() => {
  sessionStorage.clear();
});

describe("useScheduledSendHandoff", () => {
  it("keeps a message waiting for its new chat through a reload", async () => {
    const before = await freshStore();
    before.getState().hold("chat-new", "Draft the launch post");

    const after = await freshStore();

    expect(after.getState().sends["chat-new"]).toEqual({
      message: "Draft the launch post",
      sentAt: null,
    });
  });

  it("keeps a send through a reload until the request setting it settles", async () => {
    const before = await freshStore();
    before.getState().hold("chat-new", "Draft the launch post");
    before.getState().markSent("chat-new");

    const after = await freshStore();
    expect(after.getState().sends["chat-new"]?.sentAt).not.toBeNull();

    after.getState().clear("chat-new");
    expect((await freshStore()).getState().sends["chat-new"]).toBeUndefined();
  });

  it("reads a message kept before sends recorded when they were sent", async () => {
    sessionStorage.setItem(
      STORE_KEY,
      JSON.stringify({
        state: { messages: { "chat-old": "Draft the launch post" } },
        version: 0,
      }),
    );

    const store = await freshStore();

    expect(store.getState().sends["chat-old"]).toEqual({
      message: "Draft the launch post",
      sentAt: null,
    });
  });
});
