import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useNowMinute } from "../useNowMinute";
import { useNowSecond } from "../useNowSecond";

const renders: Record<string, number> = {};

function SecondLabel({ id, enabled }: { id: string; enabled?: boolean }) {
  const now = useNowSecond(enabled);
  renders[id] = (renders[id] ?? 0) + 1;
  return <span data-testid={id}>{new Date(now).toISOString()}</span>;
}

function MinuteLabel({ id }: { id: string }) {
  const now = useNowMinute();
  renders[id] = (renders[id] ?? 0) + 1;
  return <span data-testid={id}>{new Date(now).toISOString()}</span>;
}

function label(id: string) {
  return screen.getByTestId(id).textContent;
}

describe("useNowSecond", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-23T10:00:00.400Z"));
    Object.keys(renders).forEach((key) => delete renders[key]);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("gives every subscriber the same second, in the same tick", () => {
    render(
      <>
        <SecondLabel id="a" />
        <SecondLabel id="b" />
      </>,
    );
    expect(label("a")).toBe("2026-04-23T10:00:00.000Z");
    expect(label("b")).toBe("2026-04-23T10:00:00.000Z");
    expect(vi.getTimerCount()).toBe(1);

    act(() => {
      vi.advanceTimersByTime(600);
    });
    expect(label("a")).toBe("2026-04-23T10:00:01.000Z");
    expect(label("b")).toBe("2026-04-23T10:00:01.000Z");
    expect(renders).toEqual({ a: 2, b: 2 });
  });

  it("ticks on the wall-clock second boundary, once a second", () => {
    render(<SecondLabel id="a" />);
    act(() => {
      vi.advanceTimersByTime(599);
    });
    expect(label("a")).toBe("2026-04-23T10:00:00.000Z");
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(label("a")).toBe("2026-04-23T10:00:01.000Z");
    for (let i = 0; i < 3; i++) {
      act(() => {
        vi.advanceTimersByTime(1_000);
      });
    }
    expect(label("a")).toBe("2026-04-23T10:00:04.000Z");
    expect(renders.a).toBe(5);
  });

  it("stops the timer when the last subscriber unmounts", () => {
    const first = render(<SecondLabel id="a" />);
    const second = render(<SecondLabel id="b" />);
    expect(vi.getTimerCount()).toBe(1);

    first.unmount();
    expect(vi.getTimerCount()).toBe(1);
    act(() => {
      vi.advanceTimersByTime(1_000);
    });
    expect(label("b")).toBe("2026-04-23T10:00:01.000Z");

    second.unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("restarts from the current second for a new subscriber", () => {
    const first = render(<SecondLabel id="a" />);
    first.unmount();
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    render(<SecondLabel id="b" />);
    expect(label("b")).toBe("2026-04-23T10:00:05.000Z");
    expect(vi.getTimerCount()).toBe(1);
  });

  it("neither subscribes nor re-renders while disabled", () => {
    const { rerender } = render(<SecondLabel id="a" enabled={false} />);
    expect(vi.getTimerCount()).toBe(0);
    act(() => {
      vi.advanceTimersByTime(5_000);
    });
    expect(renders.a).toBe(1);

    rerender(<SecondLabel id="a" enabled />);
    expect(label("a")).toBe("2026-04-23T10:00:05.000Z");
    expect(vi.getTimerCount()).toBe(1);

    rerender(<SecondLabel id="a" enabled={false} />);
    expect(vi.getTimerCount()).toBe(0);
  });
});

describe("useNowMinute", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-23T10:00:30.000Z"));
    Object.keys(renders).forEach((key) => delete renders[key]);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("ticks once a minute on the wall-clock minute, shared by every subscriber", () => {
    const { unmount } = render(
      <>
        <MinuteLabel id="a" />
        <MinuteLabel id="b" />
      </>,
    );
    expect(label("a")).toBe("2026-04-23T10:00:00.000Z");
    expect(vi.getTimerCount()).toBe(1);

    act(() => {
      vi.advanceTimersByTime(29_999);
    });
    expect(renders).toEqual({ a: 1, b: 1 });
    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(label("a")).toBe("2026-04-23T10:01:00.000Z");
    expect(label("b")).toBe("2026-04-23T10:01:00.000Z");
    expect(renders).toEqual({ a: 2, b: 2 });

    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
