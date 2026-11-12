import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ThinkingIndicator } from "../ThinkingIndicator";

vi.mock("../../../ToolChain/SwapText", () => ({
  SwapText: ({ text }: { text: string }) => <span>{text}</span>,
}));
vi.mock("../../../DrawnLineLoader/DrawnLineLoader", () => ({
  DrawnLineLoader: () => null,
}));

const NOW = new Date("2026-04-23T10:00:00.000Z").getTime();

function advanceSeconds(seconds: number) {
  for (let i = 0; i < seconds; i++) {
    act(() => {
      vi.advanceTimersByTime(1_000);
    });
  }
}

function statusText() {
  return screen.getByRole("status").textContent;
}

describe("ThinkingIndicator", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(NOW);
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows the elapsed time from startedAtMs only after 20 s", () => {
    render(<ThinkingIndicator active startedAtMs={NOW - 18_000} />);
    expect(statusText()).toBe("Thinking…");

    advanceSeconds(1);
    expect(statusText()).toBe("Thinking…");

    advanceSeconds(1);
    expect(statusText()).toBe("Thinking…20s");

    advanceSeconds(45);
    expect(statusText()).toBe("Thinking…1m 5s");
  });

  it("runs no clock while inactive", () => {
    render(<ThinkingIndicator active={false} startedAtMs={NOW - 60_000} />);
    expect(statusText()).toBe("Thinking…");
    expect(vi.getTimerCount()).toBe(0);
  });

  it("stops its clock when it unmounts", () => {
    const { unmount } = render(<ThinkingIndicator active startedAtMs={NOW} />);
    expect(vi.getTimerCount()).toBe(1);
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });

  it("still shows a parent-owned elapsedSeconds, with no clock of its own", () => {
    const { rerender } = render(
      <ThinkingIndicator
        active
        elapsedSeconds={19}
        statusMessage="Uploading"
      />,
    );
    expect(statusText()).toBe("Uploading");
    expect(vi.getTimerCount()).toBe(0);

    rerender(
      <ThinkingIndicator
        active
        elapsedSeconds={21}
        statusMessage="Uploading"
      />,
    );
    expect(statusText()).toBe("Uploading21s");
  });
});
