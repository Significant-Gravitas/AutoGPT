import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ChainRowView } from "../ChainRowView";
import type { ChainRow } from "../helpers";

vi.mock("../SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));

const NOW = new Date("2026-04-23T10:00:00.000Z").getTime();

function advanceSeconds(seconds: number) {
  for (let i = 0; i < seconds; i++) {
    act(() => {
      vi.advanceTimersByTime(1_000);
    });
  }
}

function row(overrides: Partial<ChainRow> = {}): ChainRow {
  return {
    key: "row-1",
    category: "agent",
    tool: "run_agent",
    text: 'Running workflow "Sticker Designs"…',
    state: "running",
    ...overrides,
  };
}

describe("ChainRowView - step clock", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows how long a running step has taken once it passes 20s", () => {
    render(<ChainRowView row={row()} isLast />);
    advanceSeconds(19);
    expect(screen.queryByText("19s")).toBeNull();

    advanceSeconds(1);
    expect(screen.getByText("20s")).toBeDefined();

    advanceSeconds(105);
    expect(screen.getByText("2m 5s")).toBeDefined();
  });

  it("drops the clock once the step finishes", () => {
    const { rerender } = render(<ChainRowView row={row()} isLast />);
    advanceSeconds(30);
    expect(screen.getByText("30s")).toBeDefined();

    rerender(
      <ChainRowView
        row={row({ state: "done", text: 'Ran workflow "Sticker Designs"' })}
        isLast
      />,
    );
    expect(screen.queryByText(/^\d+s$/)).toBeNull();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("leaves thinking rows to their own narration", () => {
    render(
      <ChainRowView
        row={row({
          category: "reasoning",
          tool: undefined,
          text: "Thinking…",
          reasoningText: "Weighing the options",
        })}
        isLast
      />,
    );
    advanceSeconds(30);
    expect(screen.queryByText("30s")).toBeNull();
  });
});
