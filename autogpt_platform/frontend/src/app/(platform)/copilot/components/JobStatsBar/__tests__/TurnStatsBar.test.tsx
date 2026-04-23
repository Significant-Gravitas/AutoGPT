import { render, screen } from "@/tests/integrations/test-utils";
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

  it("prefers live elapsedSeconds over durationMs / reasoningDurationMs", () => {
    render(
      <TurnStatsBar
        turnMessages={EMPTY}
        elapsedSeconds={7}
        stats={{ durationMs: 99_000, reasoningDurationMs: 50_000 }}
      />,
    );
    expect(screen.getByText(/Thought for 7s/)).toBeDefined();
  });

  it("prefers reasoningDurationMs when the turn is finalized", () => {
    render(
      <TurnStatsBar
        turnMessages={EMPTY}
        stats={{ durationMs: 12_000, reasoningDurationMs: 3_500 }}
      />,
    );
    // 3_500ms rounds to ~4s — never falls back to whole-turn 12s.
    expect(screen.getByText(/Thought for 4s/)).toBeDefined();
  });

  it("floors reasoningDurationMs to a 1s minimum so sub-second rows still render", () => {
    render(
      <TurnStatsBar
        turnMessages={EMPTY}
        stats={{ reasoningDurationMs: 200 }}
      />,
    );
    expect(screen.getByText(/Thought for 1s/)).toBeDefined();
  });

  it("falls back to durationMs for legacy rows that have no reasoning data", () => {
    render(
      <TurnStatsBar turnMessages={EMPTY} stats={{ durationMs: 42_000 }} />,
    );
    expect(screen.getByText(/Thought for 42s/)).toBeDefined();
  });

  it("renders a local timestamp when there is no duration at all", () => {
    render(
      <TurnStatsBar
        turnMessages={EMPTY}
        stats={{ createdAt: "2026-04-23T08:32:09.000Z" }}
      />,
    );
    const labels = screen.getAllByText(
      (_, el) => !!el?.className.includes("tabular-nums"),
    );
    expect(labels.length).toBeGreaterThan(0);
  });

  it("renders malformed timestamp as its raw string (formatLocalTimestamp passthrough)", () => {
    const { container } = render(
      <TurnStatsBar turnMessages={EMPTY} stats={{ createdAt: "not-a-date" }} />,
    );
    expect(container.firstChild).not.toBeNull();
    expect(container.textContent).toContain("not-a-date");
  });

  it("renders the tooltip trigger when both duration and timestamp are present", () => {
    const { container } = render(
      <TurnStatsBar
        turnMessages={EMPTY}
        stats={{ durationMs: 5_000, createdAt: "2026-04-23T08:32:09.000Z" }}
      />,
    );
    expect(screen.getByText(/Thought for 5s/)).toBeDefined();
    expect(container.querySelector("[data-state]")).not.toBeNull();
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
