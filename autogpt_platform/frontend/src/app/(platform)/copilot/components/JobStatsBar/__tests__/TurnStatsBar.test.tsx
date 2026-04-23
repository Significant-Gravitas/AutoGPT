import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { describe, expect, it } from "vitest";
import { TurnStatsBar } from "../TurnStatsBar";

type Msg = UIMessage<unknown, UIDataTypes, UITools>;

const EMPTY: Msg[] = [];

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
