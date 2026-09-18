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

vi.mock("../../../ToolChain/ExpertCards", () => ({
  ExpertChangeGroup: ({ parts }: { parts: unknown[] }) => (
    <div data-testid="expert-group" data-count={parts.length} />
  ),
}));

describe("ChainMessageParts", () => {
  afterEach(cleanup);

  it("marks the final chain as streaming when a regular part follows it", () => {
    // Long text is a real answer — short text would fold into the chain
    // as progress narration while streaming.
    const parts = [
      {
        type: "tool-web_search",
        state: "input-available",
        toolCallId: "search",
        input: {},
      },
      { type: "text", text: "partial response. ".repeat(20) },
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

  it("renders expert approvals as one group outside the chain", () => {
    const parts = [
      {
        type: "tool-web_search",
        state: "output-available",
        toolCallId: "search",
        input: {},
        output: {},
      },
      {
        type: "tool-raise_expert",
        state: "output-available",
        toolCallId: "raise-1",
        input: {},
        output: { type: "expert_change_proposed" },
      },
      {
        type: "tool-raise_expert",
        state: "output-available",
        toolCallId: "raise-2",
        input: {},
        output: { type: "expert_change_proposed" },
      },
    ] as MessagePart[];

    render(
      <ChainMessageParts
        parts={parts}
        messageID="message"
        isCurrentlyStreaming={false}
      />,
    );

    expect(screen.getAllByTestId("tool-chain").length).toBe(1);
    expect(screen.getByTestId("expert-group").dataset.count).toBe("2");
    expect(screen.queryByTestId("message-part")).toBeNull();
  });

  it("lifts an expert approval out of the chain when it ran through run_capability", () => {
    const parts = [
      {
        type: "tool-run_capability",
        state: "input-available",
        toolCallId: "hire",
        input: { id: "tool:hire_expert", input: { name: "Researcher" } },
      },
      {
        type: "tool-run_capability",
        state: "output-available",
        toolCallId: "confirm",
        input: { id: "confirm_expert_change", input: {} },
        output: { type: "expert_change_proposed" },
      },
    ] as MessagePart[];

    render(
      <ChainMessageParts
        parts={parts}
        messageID="message"
        isCurrentlyStreaming={true}
      />,
    );

    expect(screen.queryByTestId("tool-chain")).toBeNull();
    expect(screen.getByTestId("expert-group").dataset.count).toBe("2");
  });

  it("keeps other run_capability calls in the chain", () => {
    const parts = [
      {
        type: "tool-run_capability",
        state: "output-available",
        toolCallId: "dry",
        input: { id: "tool:hire_expert", validate_only: true },
        output: {},
      },
      {
        type: "tool-run_capability",
        state: "output-available",
        toolCallId: "search",
        input: { id: "tool:web_search", input: { query: "x" } },
        output: {},
      },
    ] as MessagePart[];

    render(
      <ChainMessageParts
        parts={parts}
        messageID="message"
        isCurrentlyStreaming={false}
      />,
    );

    expect(screen.getAllByTestId("tool-chain").length).toBe(1);
    expect(screen.queryByTestId("expert-group")).toBeNull();
  });
});
