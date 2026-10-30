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

let rowCount = 0;

function row(overrides: Partial<ChainRow> = {}): ChainRow {
  rowCount += 1;
  return {
    key: `call-${rowCount}`,
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

  it("keeps counting when the row mounts again", () => {
    const step = row();
    const { unmount } = render(<ChainRowView row={step} isLast />);
    advanceSeconds(300);
    unmount();

    render(<ChainRowView row={step} isLast />);
    expect(screen.getByText("5m 0s")).toBeDefined();
  });

  it("drops the clock once the step finishes", () => {
    const step = row();
    const { rerender } = render(<ChainRowView row={step} isLast />);
    advanceSeconds(30);
    expect(screen.getByText("30s")).toBeDefined();

    rerender(
      <ChainRowView
        row={{ ...step, state: "done", text: 'Ran workflow "Sticker Designs"' }}
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

  it("leaves a call waiting for approval without a clock", () => {
    render(
      <ChainRowView
        row={row({ held: { state: "waiting", reviewId: "review-1" } })}
        isLast
      />,
    );
    advanceSeconds(30);
    expect(screen.queryByText("30s")).toBeNull();
  });

  it("keeps each running step's own count across a remount", () => {
    const first = row();
    const { unmount } = render(<ChainRowView row={first} isLast={false} />);
    advanceSeconds(60);
    const second = row();
    const { unmount: unmountSecond } = render(
      <ChainRowView row={second} isLast />,
    );
    advanceSeconds(25);
    unmount();
    unmountSecond();

    render(
      <>
        <ChainRowView row={first} isLast={false} />
        <ChainRowView row={second} isLast />
      </>,
    );
    expect(screen.getByText("1m 25s")).toBeDefined();
    expect(screen.getByText("25s")).toBeDefined();
  });

  it("keeps the ticking time out of the row's accessible name", () => {
    render(<ChainRowView row={row()} isLast />);
    advanceSeconds(20);
    expect(screen.getByText("20s").getAttribute("aria-hidden")).toBe("true");
  });
});
