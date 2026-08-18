import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { TurnStatsBar } from "../TurnStatsBar";
import { useElapsedTimer } from "../useElapsedTimer";

type Msg = UIMessage<unknown, UIDataTypes, UITools>;

const EMPTY: Msg[] = [];

interface HarnessProps {
  isRunning: boolean;
  anchor?: string | null;
}

// Renders TurnStatsBar driven by the real useElapsedTimer hook so the timer's
// behaviour is observable through the rendered "Thought for Xs" label.
function TimerHarness({ isRunning, anchor }: HarnessProps) {
  const { elapsedSeconds } = useElapsedTimer(isRunning, anchor);
  return <TurnStatsBar turnMessages={EMPTY} elapsedSeconds={elapsedSeconds} />;
}

describe("TurnStatsBar", () => {
  it("renders nothing when there is no time, no timestamp, and no counters", () => {
    const { container } = render(<TurnStatsBar turnMessages={EMPTY} />);
    expect(container.firstChild).toBeNull();
  });

  it("prefers live elapsedSeconds over the persisted durationMs", () => {
    render(
      <TurnStatsBar
        turnMessages={EMPTY}
        elapsedSeconds={7}
        stats={{ durationMs: 99_000 }}
      />,
    );
    expect(screen.getByText(/Thought for 7s/)).toBeDefined();
  });

  it("uses durationMs when the turn is finalized", () => {
    render(
      <TurnStatsBar turnMessages={EMPTY} stats={{ durationMs: 42_000 }} />,
    );
    expect(screen.getByText(/Thought for 42s/)).toBeDefined();
  });

  it("renders nothing for sub-second durations (would round to 0s)", () => {
    const { container } = render(
      <TurnStatsBar turnMessages={EMPTY} stats={{ durationMs: 400 }} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders nothing when only a timestamp is present (date is hover-only)", () => {
    const { container } = render(
      <TurnStatsBar
        turnMessages={EMPTY}
        stats={{ createdAt: "2026-04-23T08:32:09.000Z" }}
      />,
    );
    // Without a duration there's no label to hover over — render nothing.
    expect(container.firstChild).toBeNull();
  });

  it("swaps to the date on hover and reverts on mouse leave", () => {
    const { container } = render(
      <TurnStatsBar
        turnMessages={EMPTY}
        stats={{ durationMs: 5_000, createdAt: "2026-04-23T08:32:09.000Z" }}
      />,
    );
    const label = container.querySelector("div.mt-2 > span") as HTMLElement;
    expect(label.textContent).toMatch(/Thought for 5s/);
    fireEvent.mouseEnter(label);
    expect(label.textContent).not.toMatch(/Thought for/);
    expect(label.textContent).toMatch(/2026/);
    fireEvent.mouseLeave(label);
    expect(label.textContent).toMatch(/Thought for 5s/);
  });

  it("renders work-done counters from assistant tool parts", () => {
    const messages: Msg[] = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          {
            type: "tool-run_agent",
            toolCallId: "t1",
            state: "output-available",
            input: {},
            output: {},
          },
          {
            type: "tool-run_agent",
            toolCallId: "t2",
            state: "output-available",
            input: {},
            output: {},
          },
          {
            type: "tool-run_block",
            toolCallId: "t3",
            state: "output-available",
            input: {},
            output: {},
          },
        ] as Msg["parts"],
      },
    ];
    const { container } = render(
      <TurnStatsBar turnMessages={messages} stats={{ durationMs: 4_000 }} />,
    );
    expect(screen.getByText(/Thought for 4s/)).toBeDefined();
    const bar = container.querySelector("div.mt-2");
    expect(bar?.textContent).toMatch(/2\s*agents run/);
    expect(bar?.textContent).toMatch(/1\s*action/);
  });
});

describe("TurnStatsBar — live elapsed timer", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-23T10:00:00.000Z"));
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  it("ticks once per second while running so the label reflects real time", () => {
    const { container } = render(<TimerHarness isRunning />);
    // Elapsed starts at 0, which renders nothing.
    expect(container.firstChild).toBeNull();
    act(() => {
      vi.advanceTimersByTime(3000);
    });
    expect(screen.getByText(/Thought for 3s/)).toBeDefined();
  });

  it("freezes the displayed elapsed time when isRunning flips to false", () => {
    const { rerender } = render(<TimerHarness isRunning />);
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByText(/Thought for 2s/)).toBeDefined();

    rerender(<TimerHarness isRunning={false} />);
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    // No further ticks — label stays at the last reading.
    expect(screen.getByText(/Thought for 2s/)).toBeDefined();
  });

  it("anchors to an ISO timestamp so a fresh mount reflects real elapsed time", () => {
    const anchor = new Date("2026-04-23T09:59:45.000Z").toISOString();
    render(<TimerHarness isRunning anchor={anchor} />);
    expect(screen.getByText(/Thought for 15s/)).toBeDefined();
    act(() => {
      vi.advanceTimersByTime(5000);
    });
    expect(screen.getByText(/Thought for 20s/)).toBeDefined();
  });

  it("clamps a future-dated anchor to zero rather than a negative reading", () => {
    const anchor = new Date("2026-04-23T10:00:10.000Z").toISOString();
    const { container } = render(<TimerHarness isRunning anchor={anchor} />);
    // Elapsed clamped to 0 → TurnStatsBar renders nothing.
    expect(container.firstChild).toBeNull();
  });

  it("falls back to mount-time counting when the anchor is invalid", () => {
    const { container } = render(
      <TimerHarness isRunning anchor="not-a-date" />,
    );
    expect(container.firstChild).toBeNull();
    act(() => {
      vi.advanceTimersByTime(4000);
    });
    expect(screen.getByText(/Thought for 4s/)).toBeDefined();
  });

  it("re-syncs when a late-arriving anchor replaces the previous one mid-run", () => {
    const { rerender } = render(<TimerHarness isRunning anchor={null} />);
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(screen.getByText(/Thought for 2s/)).toBeDefined();

    rerender(
      <TimerHarness
        isRunning
        anchor={new Date("2026-04-23T09:59:00.000Z").toISOString()}
      />,
    );
    // Clock is at 10:00:02, anchor is 60s earlier, so elapsed jumps to 62s.
    expect(screen.getByText(/Thought for 1m 2s/)).toBeDefined();
  });
});
