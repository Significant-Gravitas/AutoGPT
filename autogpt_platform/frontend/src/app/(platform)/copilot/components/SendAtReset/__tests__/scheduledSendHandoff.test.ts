import { beforeEach, describe, expect, it, vi } from "vitest";

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

    expect(after.getState().take("chat-new")).toBe("Draft the launch post");
    expect(after.getState().take("chat-new")).toBeNull();
  });

  it("hands a message over once, even after a reload", async () => {
    const before = await freshStore();
    before.getState().hold("chat-new", "Draft the launch post");
    expect(before.getState().take("chat-new")).toBe("Draft the launch post");

    const after = await freshStore();

    expect(after.getState().take("chat-new")).toBeNull();
  });
});
