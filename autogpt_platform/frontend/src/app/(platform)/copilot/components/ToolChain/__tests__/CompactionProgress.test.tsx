import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CompactionProgress } from "../CompactionProgress";

function fillWidth(container: HTMLElement) {
  const bar = container.querySelector("[role=progressbar] > div");
  if (!bar) return null;
  return parseFloat((bar as HTMLElement).style.width);
}

function tick(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

describe("CompactionProgress", () => {
  beforeEach(() => vi.useFakeTimers());

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("stays hidden for a row replayed from history", () => {
    const { container } = render(<CompactionProgress done />);

    expect(fillWidth(container)).toBeNull();
  });

  it("advances while running but never reaches the end", () => {
    const { container } = render(<CompactionProgress done={false} />);
    expect(screen.getByRole("progressbar")).toBeDefined();

    tick(5_000);
    const early = fillWidth(container)!;
    expect(early).toBeGreaterThan(0);

    tick(120_000);
    expect(fillWidth(container)!).toBeGreaterThan(early);
    expect(fillWidth(container)!).toBeLessThan(100);
  });

  it("sprints to the end when compaction finishes, then unmounts", () => {
    const { container, rerender } = render(<CompactionProgress done={false} />);
    tick(5_000);

    rerender(<CompactionProgress done />);
    tick(700);
    expect(fillWidth(container)).toBe(100);

    tick(1_000);
    expect(fillWidth(container)).toBeNull();
  });
});
