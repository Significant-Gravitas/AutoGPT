import { makeTask } from "@/app/(platform)/copilot/components/EmptySession/__tests__/taskFixtures";
import type { Task } from "@/app/api/__generated__/models/task";
import { act, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { buildBoardColumns } from "../helpers";
import {
  QUICK_STARTS,
  formatStartTime,
  formatStartsIn,
  fromLocalInputValue,
  isMissed,
  isScheduled,
  rhythmLine,
  scheduleStateOf,
  startsToday,
  toLocalInputValue,
} from "../schedule";
import { useStartCountdown } from "../useStartCountdown";

// Local times, so the day boundaries hold in any time zone.
const NOW = new Date(2026, 10, 15, 10, 0, 0).getTime();
const MIN = 60_000;

function clock(at: number) {
  return new Date(at).toLocaleTimeString(undefined, {
    hour: "numeric",
    minute: "2-digit",
  });
}

function task(extra: Partial<Task>): Task {
  return { ...makeTask("t", "Research AutoGPT", "TODO"), ...extra };
}

describe("schedule state", () => {
  it("calls a to-do with a start time scheduled and a passed one missed", () => {
    const scheduled = task({ start_at: new Date(NOW + 5 * MIN) });
    const missed = task({ start_at: null, missed_start: true });
    const plain = task({});
    const started = task({ status: "IN_PROGRESS", start_at: new Date(NOW) });

    expect([isScheduled(scheduled), isMissed(scheduled)]).toEqual([
      true,
      false,
    ]);
    expect([isScheduled(missed), isMissed(missed)]).toEqual([false, true]);
    expect(scheduleStateOf(plain)).toBeNull();
    expect(scheduleStateOf(started)).toBeNull();
    expect(scheduleStateOf(scheduled)).toBe("scheduled");
    expect(scheduleStateOf(missed)).toBe("missed");
  });
});

describe("formatStartsIn", () => {
  it("counts seconds, then minutes, then names the time and day", () => {
    const later = new Date(2026, 10, 15, 15, 30).getTime();
    const tomorrow = new Date(2026, 10, 16, 9, 0).getTime();

    expect(formatStartsIn(NOW + 40_000, NOW)).toBe("Starts in 40s");
    expect(formatStartsIn(NOW + 4 * MIN, NOW)).toBe("Starts in 4 min");
    expect(formatStartsIn(NOW + 3.5 * MIN, NOW)).toBe("Starts in 4 min");
    expect(formatStartsIn(later, NOW)).toBe(`Starts at ${clock(later)}`);
    expect(formatStartsIn(tomorrow, NOW)).toBe(
      `Starts tomorrow ${clock(tomorrow)}`,
    );
    expect(formatStartsIn(NOW - 1000, NOW)).toBe("Starting now");
    expect(formatStartsIn(null, NOW)).toBe("");
  });

  it("gives the exact local time, with the day when it is not today", () => {
    const tomorrow = new Date(2026, 10, 16, 9, 0).getTime();
    expect(formatStartTime(NOW + 5 * MIN, NOW)).toBe(clock(NOW + 5 * MIN));
    expect(formatStartTime(tomorrow, NOW)).toBe(`Tomorrow, ${clock(tomorrow)}`);
  });
});

describe("start time helpers", () => {
  it("finds the tasks starting before today ends", () => {
    expect(startsToday(task({ start_at: new Date(NOW + 60 * MIN) }), NOW)).toBe(
      true,
    );
    const nextDay = new Date(2026, 10, 16, 0, 30);
    expect(startsToday(task({ start_at: nextDay }), NOW)).toBe(false);
    expect(startsToday(task({}), NOW)).toBe(false);
  });

  it("offers in 5 min, in 1 hour and tomorrow at nine", () => {
    const [five, hour, tomorrow] = QUICK_STARTS.map((q) => q.at(NOW));
    expect(five.getTime()).toBe(NOW + 5 * MIN);
    expect(hour.getTime()).toBe(NOW + 60 * MIN);
    expect(tomorrow.getTime()).toBe(new Date(2026, 10, 16, 9, 0).getTime());
  });

  it("round-trips the date-time input's local value", () => {
    const at = new Date(2026, 10, 15, 10, 5);
    expect(toLocalInputValue(at)).toBe("2026-11-15T10:05");
    expect(fromLocalInputValue("2026-11-15T10:05")?.getTime()).toBe(
      at.getTime(),
    );
    expect(fromLocalInputValue("")).toBeNull();
  });
});

describe("buildBoardColumns", () => {
  it("adds a Scheduled lane, soonest first, and puts a missed start in Needs you", () => {
    const later = task({ id: "later", start_at: new Date(NOW + 60 * MIN) });
    const soon = task({ id: "soon", start_at: new Date(NOW + 5 * MIN) });
    const missed = task({ id: "missed", missed_start: true });
    const plain = task({ id: "plain" });

    const columns = buildBoardColumns([later, plain, missed, soon]);

    expect(columns.map((c) => c.label)).toEqual([
      "Needs you",
      "Scheduled",
      "Working",
      "Done",
      "Cancelled",
    ]);
    expect(columns[0].tasks.map((t) => t.id)).toEqual(["missed"]);
    expect(columns[1].tasks.map((t) => t.id)).toEqual(["soon", "later"]);
    expect(columns[2].tasks.map((t) => t.id)).toEqual(["plain"]);
  });

  it("leaves the Scheduled lane out when nothing waits for a start time", () => {
    const columns = buildBoardColumns([task({ id: "plain" })]);
    expect(columns.map((c) => c.id)).not.toContain("scheduled");
  });
});

function Countdown({ startAt }: { startAt: Date | null }) {
  const countdown = useStartCountdown(startAt);
  return <span data-testid="countdown">{countdown?.label ?? "none"}</span>;
}

describe("useStartCountdown", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  function label() {
    return screen.getByTestId("countdown").textContent;
  }

  it("ticks every 30 s while far off and every second in the last minute", () => {
    render(<Countdown startAt={new Date(NOW + 2 * MIN)} />);
    expect(label()).toBe("Starts in 2 min");

    act(() => {
      vi.advanceTimersByTime(30_000);
    });
    expect(label()).toBe("Starts in 2 min");

    act(() => {
      vi.advanceTimersByTime(45_000);
    });
    expect(label()).toBe("Starts in 45s");

    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(label()).toBe("Starts in 44s");

    act(() => {
      vi.advanceTimersByTime(44_000);
    });
    expect(label()).toBe("Starting now");
  });

  it("works from the start time, so a fresh mount shows the same time left", () => {
    const startAt = new Date(NOW + 10 * MIN);
    const first = render(<Countdown startAt={startAt} />);
    act(() => {
      vi.advanceTimersByTime(3 * MIN);
    });
    first.unmount();

    render(<Countdown startAt={startAt} />);
    expect(label()).toBe("Starts in 7 min");
  });

  it("has no timer when there is no start time, and clears its timers on unmount", () => {
    const idle = render(<Countdown startAt={null} />);
    expect(label()).toBe("none");
    expect(vi.getTimerCount()).toBe(0);
    idle.unmount();

    const live = render(<Countdown startAt={new Date(NOW + 30_000)} />);
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    live.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("repeating tasks", () => {
  const repeat = { crons: ["0 8 * * *"], label: "Every day at 08:00" };

  it("waits in Scheduled between runs and when paused, not while it runs", () => {
    const waiting = task({ id: "a", start_at: new Date(NOW + MIN), repeat });
    const paused = task({
      id: "b",
      start_at: null,
      repeat: { ...repeat, paused: true },
    });
    const running = task({ id: "c", status: "IN_PROGRESS", repeat });

    expect(scheduleStateOf(waiting)).toBe("repeating");
    expect(scheduleStateOf(paused)).toBe("paused");
    expect(scheduleStateOf(running)).toBeNull();
    const columns = buildBoardColumns([running, paused, waiting]);
    const lane = columns.find((column) => column.id === "scheduled");
    // Soonest next run first; a paused one has none, so it goes last.
    expect(lane?.tasks.map((t) => t.id)).toEqual(["a", "b"]);
  });

  it("says its rhythm and its next run, or that it is paused", () => {
    expect(rhythmLine("Every day at 08:00", false, "Starts in 4 min")).toBe(
      "Every day at 08:00 · next run in 4 min",
    );
    expect(
      rhythmLine("Every day at 08:00", false, "Starts tomorrow 8:00 AM"),
    ).toBe("Every day at 08:00 · next run tomorrow 8:00 AM");
    expect(rhythmLine("Every day at 08:00", true, "Starts in 4 min")).toBe(
      "Every day at 08:00 · Paused",
    );
  });
});
