import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

vi.mock("../../../ToolChain/ToolChain", () => ({
  ToolChain: ({ isStreaming }: { isStreaming: boolean }) => (
    <div data-testid="tool-chain" data-streaming={String(isStreaming)} />
  ),
}));

vi.mock("../MessagePartRenderer", () => ({
  MessagePartRenderer: () => <div data-testid="message-part" />,
}));

describe("ChainMessageParts", () => {
  afterEach(cleanup);

  it("marks the final chain as streaming when a regular part follows it", () => {
    const parts = [
      {
        type: "tool-web_search",
        state: "input-available",
        toolCallId: "search",
        input: {},
      },
      { type: "text", text: "partial response" },
    ] as MessagePart[];

    render(
      <ChainMessageParts
        parts={parts}
        messageID="message"
        isCurrentlyStreaming
      />,
    );

    expect(screen.getByTestId("tool-chain").dataset.streaming).toBe("true");
    expect(screen.getByTestId("message-part")).toBeDefined();
  });
});
