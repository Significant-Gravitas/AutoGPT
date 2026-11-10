import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen } from "@testing-library/react";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

// Every chain row draws its icon once per render, so counting icon renders
// per row key counts which rows rendered.
const rowRenders = new Map<string, number>();

vi.mock("../../../ToolChain/RowIcon", () => ({
  RowIcon: ({ row }: { row: { key: string } }) => {
    rowRenders.set(row.key, (rowRenders.get(row.key) ?? 0) + 1);
    return null;
  },
  ProviderIcon: () => null,
}));

vi.mock("../MessagePartRenderer", () => ({
  MessagePartRenderer: ({ part }: { part: { type: string; text?: string } }) =>
    part.type === "text" ? <p data-testid="answer">{part.text}</p> : null,
}));

function search(id: string, state: string, query: string): MessagePart {
  return {
    type: "tool-web_search",
    toolCallId: id,
    state,
    input: { query },
    output: state === "output-available" ? { results: [query] } : undefined,
  } as MessagePart;
}

// What the SDK hands over on each flush: every part copied one level deep,
// its input and output shared with the previous flush.
function nextFlush(parts: MessagePart[]): MessagePart[] {
  return parts.map((part) => ({ ...part }) as MessagePart);
}

function grow(parts: MessagePart[], token: string): MessagePart[] {
  return nextFlush(parts).map((part) =>
    part.type === "text" ? { ...part, text: part.text + token } : part,
  );
}

// One client and one send handler for the whole file, as on the page: the
// shared test wrapper builds a new query client on every rerender, which
// would re-render every row that reads a query.
const queryClient = new QueryClient();
const onSend = vi.fn();

function message(parts: MessagePart[]) {
  return (
    <QueryClientProvider client={queryClient}>
      <CopilotChatActionsProvider onSend={onSend}>
        <ChainMessageParts parts={parts} messageID="a1" isCurrentlyStreaming />
      </CopilotChatActionsProvider>
    </QueryClientProvider>
  );
}

describe("ChainMessageParts — memoised chain rows", () => {
  beforeEach(() => {
    rowRenders.clear();
  });
  afterEach(cleanup);

  const settled = [
    search("call-1", "output-available", "hotels"),
    search("call-2", "output-available", "flights"),
    { type: "text", text: "Here is what I found" } as MessagePart,
  ];

  it("leaves settled rows alone while the answer below them streams", () => {
    const { rerender } = render(message(settled));
    expect([...rowRenders.keys()].sort()).toEqual(["call-1", "call-2"]);
    rowRenders.clear();

    let parts = settled;
    for (const token of [" for", " your", " trip."]) {
      parts = grow(parts, token);
      rerender(message(parts));
    }

    expect(screen.getByTestId("answer").textContent).toBe(
      "Here is what I found for your trip.",
    );
    expect(rowRenders.size).toBe(0);
  });

  it("re-renders only the row whose call finished", () => {
    const live = [
      search("call-1", "output-available", "hotels"),
      search("call-2", "input-available", "flights"),
    ];
    const { rerender } = render(message(live));
    rowRenders.clear();

    rerender(
      message([
        { ...live[0] } as MessagePart,
        search("call-2", "output-available", "flights"),
      ]),
    );

    expect([...rowRenders.keys()]).toEqual(["call-2"]);
  });
});
