import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { ChatMessage } from "../../helpers";
import { NewChatView } from "../NewChatView";

vi.mock("../../../components/JobStatsBar/useElapsedTimer", () => ({
  useElapsedTimer: () => ({ elapsedSeconds: 3 }),
}));

describe("NewChatView", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("renders user messages and settled assistant text", () => {
    const messages: ChatMessage[] = [
      {
        id: "user-1",
        role: "user",
        parts: [{ type: "text", text: "Research EV sales" }],
      },
      {
        id: "assistant-1",
        role: "assistant",
        parts: [
          { type: "reasoning", text: "Hidden reasoning" },
          { type: "text", text: "Research complete" },
        ],
      },
    ];

    render(
      <NewChatView messages={messages} status="ready" statusMessage={null} />,
    );

    expect(screen.getByText("Research EV sales")).toBeDefined();
    expect(screen.getByText("Research complete")).toBeDefined();
    expect(screen.getByText("Hidden reasoning")).toBeDefined();
  });

  it("types the live assistant tail and shows streaming status", () => {
    const messages: ChatMessage[] = [
      {
        id: "assistant-1",
        role: "assistant",
        parts: [{ type: "text", text: "Live response" }],
      },
    ];

    render(
      <NewChatView
        messages={messages}
        status="streaming"
        statusMessage="Analyzing"
      />,
    );
    act(() => vi.advanceTimersByTime(100));

    expect(screen.getByText("Live response")).toBeDefined();
    expect(screen.getByText("Analyzing")).toBeDefined();
  });
});
