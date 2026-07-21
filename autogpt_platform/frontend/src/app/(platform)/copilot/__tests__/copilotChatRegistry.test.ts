import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@ai-sdk/react", () => ({
  Chat: vi.fn().mockImplementation(function (
    this: { options: unknown },
    options: unknown,
  ) {
    this.options = options;
  }),
}));

vi.mock("../copilotStreamTransport", () => ({
  createCopilotTransport: vi.fn(() => ({})),
}));

import { Chat } from "@ai-sdk/react";
import {
  getOrCreateCopilotChatRuntime,
  resetCopilotChatRegistry,
} from "../copilotChatRegistry";

const mockChat = vi.mocked(Chat);

function chatOptions() {
  const call = mockChat.mock.calls.at(-1);
  if (!call) throw new Error("Chat was not constructed");
  return call[0] as {
    onData?: (dataPart: { type: string; data?: unknown }) => void;
  };
}

describe("copilotChatRegistry onData wiring", () => {
  beforeEach(() => {
    resetCopilotChatRegistry();
    mockChat.mockClear();
  });

  it("forwards chat data parts to the registered handler", () => {
    const runtime = getOrCreateCopilotChatRuntime("s1");
    const handler = vi.fn();
    runtime.onData = handler;

    const part = { type: "data-mode-changed", data: { mode: "fast" } };
    chatOptions().onData?.(part);

    expect(handler).toHaveBeenCalledWith(part);
    expect(runtime.onData).toBe(handler);
  });

  it("ignores data parts when no handler is registered", () => {
    getOrCreateCopilotChatRuntime("s1");
    expect(() =>
      chatOptions().onData?.({ type: "data-mode-changed" }),
    ).not.toThrow();
  });

  it("stops forwarding after the handler is cleared", () => {
    const runtime = getOrCreateCopilotChatRuntime("s1");
    const handler = vi.fn();
    runtime.onData = handler;
    runtime.onData = undefined;

    chatOptions().onData?.({ type: "data-mode-changed" });

    expect(handler).not.toHaveBeenCalled();
    expect(runtime.onData).toBeUndefined();
  });

  it("reuses the runtime (and its handler slot) per session", () => {
    const first = getOrCreateCopilotChatRuntime("s1");
    const second = getOrCreateCopilotChatRuntime("s1");
    expect(second).toBe(first);
    expect(mockChat).toHaveBeenCalledTimes(1);
  });
});
