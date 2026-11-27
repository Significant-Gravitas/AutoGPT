import { hydrateAcrossClocks } from "@/tests/integrations/hydration";
import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { RelativeTime } from "../RelativeTime";

let parentRenders = 0;

function Row({
  id,
  date,
  variant,
}: {
  id: string;
  date: Date | string | null;
  variant?: "long" | "short";
}) {
  parentRenders += 1;
  return (
    <span data-testid={id}>
      <RelativeTime date={date} variant={variant} />
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

  it("shows it in short when asked", () => {
    render(
      <>
        <Row id="a" date="2026-04-23T09:55:30.000Z" variant="short" />
        <Row id="b" date="2026-04-21T10:00:30.000Z" variant="short" />
      </>,
    );
    expect(label("a")).toBe("5m ago");
    expect(label("b")).toBe("2d ago");
  });

  it("shows nothing for a missing or unreadable date", () => {
    render(
      <>
        <Row id="a" date={null} />
        <Row id="b" date="not a date" />
      </>,
    );
    expect(label("a")).toBe("");
    expect(label("b")).toBe("");
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

describe("RelativeTime in a server-rendered page", () => {
  const submitted = new Date("2026-04-23T23:55:30.000Z");
  const serverNow = new Date("2026-04-23T23:56:00.000Z");
  // The visitor's clock, five minutes ahead of the server's.
  const browserNow = new Date("2026-04-24T00:01:00.000Z");

  beforeEach(() => {
    vi.useFakeTimers({ toFake: ["Date"] });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("hydrates without a mismatch when the browser's clock is ahead of the server's", () => {
    const { hydrationErrors } = hydrateAcrossClocks({
      ui: <Row id="a" date={submitted} />,
      serverNow,
      browserNow,
    });

    expect(hydrationErrors).toEqual([]);
  });

  it("serves the date itself, in UTC, then the browser's relative time once hydrated", () => {
    const { serverText } = hydrateAcrossClocks({
      ui: <Row id="a" date={submitted} variant="short" />,
      serverNow,
      browserNow,
    });

    expect(serverText).toBe("Apr 23, 2026");
    expect(label("a")).toBe("5m ago");
  });
});
