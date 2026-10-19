import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useElapsedAnchor, useElapsedTimer } from "../useElapsedTimer";

const NOW = new Date("2026-04-23T10:00:00.000Z").getTime();
const renders: Record<string, number> = {};

interface HarnessProps {
  id: string;
  isRunning: boolean;
  anchor?: string | null;
}

function TimerHarness({ id, isRunning, anchor }: HarnessProps) {
  const { elapsedSeconds } = useElapsedTimer(isRunning, anchor);
  renders[id] = (renders[id] ?? 0) + 1;
  return <span data-testid={id}>{elapsedSeconds}</span>;
}

function AnchorHarness({ id, isRunning, anchor }: HarnessProps) {
  const { startedAtMs, frozenSeconds } = useElapsedAnchor(isRunning, anchor);
  renders[id] = (renders[id] ?? 0) + 1;
  return (
    <span data-testid={id}>{`${startedAtMs ?? "none"}|${frozenSeconds}`}</span>
  );
}

function text(id: string) {
  return screen.getByTestId(id).textContent;
}

function advanceSeconds(seconds: number) {
  for (let i = 0; i < seconds; i++) {
    act(() => {
      vi.advanceTimersByTime(1_000);
    });
  }
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(NOW);
  Object.keys(renders).forEach((key) => delete renders[key]);
});
afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("useElapsedTimer", () => {
  it("ticks every running timer from one shared clock, in the same tick", () => {
    render(
      <>
        <TimerHarness id="a" isRunning />
        <TimerHarness
          id="b"
          isRunning
          anchor={new Date(NOW - 30_000).toISOString()}
        />
      </>,
    );
    expect(vi.getTimerCount()).toBe(1);

    advanceSeconds(2);
    expect(text("a")).toBe("2");
    expect(text("b")).toBe("32");
    expect(renders).toEqual({ a: 3, b: 3 });
  });

  it("releases the shared clock once no timer is running", () => {
    const { rerender } = render(<TimerHarness id="a" isRunning />);
    expect(vi.getTimerCount()).toBe(1);
    rerender(<TimerHarness id="a" isRunning={false} />);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("useElapsedAnchor", () => {
  it("holds the run's start without re-rendering every second", () => {
    render(<AnchorHarness id="a" isRunning />);
    expect(text("a")).toBe(`${NOW}|0`);

    advanceSeconds(5);
    expect(renders.a).toBe(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("starts from the server anchor and re-syncs when a later one arrives", () => {
    const { rerender } = render(
      <AnchorHarness id="a" isRunning anchor={null} />,
    );
    expect(text("a")).toBe(`${NOW}|0`);

    const anchor = new Date(NOW - 60_000).toISOString();
    rerender(<AnchorHarness id="a" isRunning anchor={anchor} />);
    expect(text("a")).toBe(`${NOW - 60_000}|0`);
  });

  it("freezes the elapsed reading when the run stops and keeps it until the next one", () => {
    const { rerender } = render(<AnchorHarness id="a" isRunning />);
    advanceSeconds(7);

    rerender(<AnchorHarness id="a" isRunning={false} />);
    expect(text("a")).toBe("none|7");

    advanceSeconds(3);
    rerender(<AnchorHarness id="a" isRunning />);
    expect(text("a")).toBe(`${NOW + 10_000}|7`);

    advanceSeconds(2);
    rerender(<AnchorHarness id="a" isRunning={false} />);
    expect(text("a")).toBe("none|2");
  });

  it("freezes the exact whole seconds of a run that did not start on a second", () => {
    vi.setSystemTime(NOW + 700);
    const { rerender } = render(<AnchorHarness id="a" isRunning />);

    act(() => {
      vi.advanceTimersByTime(5_100);
    });
    rerender(<AnchorHarness id="a" isRunning={false} />);
    expect(text("a")).toBe("none|5");
  });

  it("keeps the run's start when the anchor drops out mid-run", () => {
    const anchor = new Date(NOW - 60_000).toISOString();
    const { rerender } = render(
      <AnchorHarness id="a" isRunning anchor={anchor} />,
    );
    expect(text("a")).toBe(`${NOW - 60_000}|0`);

    advanceSeconds(5);
    rerender(<AnchorHarness id="a" isRunning anchor={null} />);
    expect(text("a")).toBe(`${NOW - 60_000}|0`);

    rerender(<AnchorHarness id="a" isRunning={false} anchor={null} />);
    expect(text("a")).toBe("none|65");
  });
});
