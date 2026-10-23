import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RelativeTime } from "../RelativeTime";

let parentRenders = 0;

function Row({ id, date }: { id: string; date: Date }) {
  parentRenders += 1;
  return (
    <span data-testid={id}>
      <RelativeTime date={date} />
    </span>
  );
}

function label(id: string) {
  return screen.getByTestId(id).textContent;
}

describe("RelativeTime", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-04-23T10:00:30.000Z"));
    parentRenders = 0;
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows the time relative to now, as formatDistanceToNow does", () => {
    render(<Row id="a" date={new Date("2026-04-23T09:55:30.000Z")} />);
    expect(label("a")).toBe("5 minutes ago");
  });

  it("updates on the minute without its parent re-rendering", () => {
    vi.setSystemTime(new Date("2026-04-23T10:00:20.000Z"));
    render(<Row id="a" date={new Date("2026-04-23T10:00:00.000Z")} />);
    expect(label("a")).toBe("less than a minute ago");

    act(() => {
      vi.advanceTimersByTime(39_999);
    });
    expect(label("a")).toBe("less than a minute ago");

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(label("a")).toBe("1 minute ago");
    expect(parentRenders).toBe(1);
  });

  it("shares one timer between every label on the page", () => {
    render(
      <>
        <Row id="a" date={new Date("2026-04-23T10:00:00.000Z")} />
        <Row id="b" date={new Date("2026-04-23T09:00:00.000Z")} />
        <Row id="c" date={new Date("2026-04-22T10:00:00.000Z")} />
      </>,
    );
    expect(vi.getTimerCount()).toBe(1);
    expect(label("b")).toBe("about 1 hour ago");
    expect(label("c")).toBe("1 day ago");
  });
});
