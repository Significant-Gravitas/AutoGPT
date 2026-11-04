import { cleanup, render, screen } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, describe, expect, it, vi } from "vitest";
import heldConfirm from "../../../ToolChain/__tests__/heldConfirm.json";
import { HeldOutcomesContext } from "../../HeldOutcomesContext";
import { getHeldOutcomes } from "../../heldCallRows";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

const toolChainParts: unknown[] = [];

vi.mock("../../../ToolChain/ToolChain", () => ({
  ToolChain: ({
    isStreaming,
    parts,
  }: {
    isStreaming: boolean;
    parts: unknown;
  }) => {
    toolChainParts.push(parts);
    return (
      <div data-testid="tool-chain" data-streaming={String(isStreaming)} />
    );
  },
}));

vi.mock("../MessagePartRenderer", () => ({
  MessagePartRenderer: () => <div data-testid="message-part" />,
}));

vi.mock("../../../ToolChain/ExpertCards", () => ({
  ExpertChangeGroup: ({
    parts,
  }: {
    parts: { output?: { type?: string } }[];
  }) => (
    <div
      data-testid="expert-group"
      data-count={parts.length}
      data-outputs={parts.map((part) => part.output?.type).join(",")}
    />
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

  it("hands each chain the same parts while the message's parts are unchanged", () => {
    const parts = [
      {
        type: "tool-web_search",
        state: "output-available",
        toolCallId: "search",
        input: {},
        output: {},
      },
    ] as MessagePart[];
    toolChainParts.length = 0;

    const { rerender } = render(
      <ChainMessageParts
        parts={parts}
        messageID="message"
        isCurrentlyStreaming
      />,
    );
    rerender(
      <ChainMessageParts
        parts={parts}
        messageID="message"
        isCurrentlyStreaming={false}
      />,
    );

    expect(toolChainParts).toHaveLength(2);
    expect(toolChainParts[1]).toBe(toolChainParts[0]);
  });

  it("redraws a held card outside the chain when its result lands, and keeps the chain's parts", () => {
    const parts = [
      {
        type: "tool-web_search",
        state: "output-available",
        toolCallId: "search",
        input: {},
        output: {},
      },
      heldConfirm.part,
    ] as MessagePart[];
    const turn: Message = { id: "a1", role: "assistant", parts };
    const lateResult: Message = {
      id: "u1",
      role: "user",
      metadata: heldConfirm.result.metadata,
      parts: [{ type: "text", text: heldConfirm.result.content }],
    };
    toolChainParts.length = 0;

    const { rerender } = render(
      <HeldOutcomesContext.Provider value={getHeldOutcomes([turn])}>
        <ChainMessageParts
          parts={parts}
          messageID="a1"
          isCurrentlyStreaming={false}
        />
      </HeldOutcomesContext.Provider>,
    );
    expect(screen.getByTestId("expert-group").dataset.outputs).toBe(
      "approval_required",
    );

    rerender(
      <HeldOutcomesContext.Provider value={getHeldOutcomes([turn, lateResult])}>
        <ChainMessageParts
          parts={parts}
          messageID="a1"
          isCurrentlyStreaming={false}
        />
      </HeldOutcomesContext.Provider>,
    );

    expect(screen.getByTestId("expert-group").dataset.outputs).toBe(
      "expert_change_applied",
    );
    expect(toolChainParts).toHaveLength(2);
    expect(toolChainParts[1]).toBe(toolChainParts[0]);
  });
});

type Message = UIMessage<unknown, UIDataTypes, UITools>;
