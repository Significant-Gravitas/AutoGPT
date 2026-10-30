import {
  act,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { FROZEN_NOTE } from "../../../stream/turnTail";
import { CANCELLED_MARKER } from "../../../useCopilotStop";
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

  it("reads the saved durationMs over the live count once the server has it", () => {
    render(
      <TurnStatsBar
        turnMessages={EMPTY}
        elapsedSeconds={67}
        stats={{ durationMs: 30_000 }}
      />,
    );
    expect(screen.getByText("Thought for 30s")).toBeDefined();
  });

  it("keeps the live count off the label once a zero duration is saved", () => {
    const { container } = render(
      <TurnStatsBar
        turnMessages={EMPTY}
        elapsedSeconds={67}
        stats={{ durationMs: 0 }}
      />,
    );
    // A reload has no live count, so it shows no label either.
    expect(container.firstChild).toBeNull();
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
    expect(screen.getByText("Done in 4s")).toBeDefined();
    const bar = container.querySelector("div.mt-2");
    expect(bar?.textContent).toMatch(/2\s*workflows run/);
    expect(bar?.textContent).toMatch(/1\s*action/);
  });
});

function assistant(parts: unknown[]): Msg {
  return { id: "a1", role: "assistant", parts: parts as Msg["parts"] };
}

const reply = { type: "text", text: "Here are your designs." };
const ranWorkflow = {
  type: "tool-run_agent",
  toolCallId: "t1",
  state: "output-available",
  input: {},
  output: {},
};

describe("TurnStatsBar — how the turn ended", () => {
  it("says the turn is done, with a check and the time it took", () => {
    render(
      <TurnStatsBar
        turnMessages={[assistant([reply])]}
        stats={{ durationMs: 134_000 }}
      />,
    );
    const line = screen.getByTestId("turn-end-line");
    expect(line.textContent).toBe("Done in 2m 14s");
    expect(line.querySelector("svg")).not.toBeNull();
  });

  it("says done for a turn that ended in tools, with what it produced", () => {
    render(
      <TurnStatsBar
        turnMessages={[
          assistant([ranWorkflow, { ...ranWorkflow, toolCallId: "t2" }]),
        ]}
        elapsedSeconds={95}
      />,
    );
    expect(screen.getByTestId("turn-end-line").textContent).toMatch(
      /^Done in 1m 35s·\s*2 workflows run$/,
    );
  });

  it("says done without a time when only the work done is known", () => {
    render(<TurnStatsBar turnMessages={[assistant([ranWorkflow])]} />);
    expect(screen.getByTestId("turn-end-line").textContent).toMatch(
      /^Done·\s*1 workflow run$/,
    );
  });

  it("says stopped, not done, when the user stopped the turn", () => {
    render(
      <TurnStatsBar
        turnMessages={[
          assistant([ranWorkflow, { type: "text", text: CANCELLED_MARKER }]),
        ]}
        elapsedSeconds={8}
      />,
    );
    const line = screen.getByTestId("turn-end-line");
    expect(line.textContent).toMatch(/^Stopped after 8s/);
    expect(line.querySelector("svg")).toBeNull();
  });

  it("says stopped for the stop note the server saves", () => {
    render(
      <TurnStatsBar
        turnMessages={[
          assistant([
            reply,
            {
              type: "text",
              text: "[__COPILOT_SYSTEM_e3b0__] Execution stopped by user",
            },
          ]),
        ]}
        stats={{ durationMs: 8_000 }}
      />,
    );
    expect(screen.getByTestId("turn-end-line").textContent).toBe(
      "Stopped after 8s",
    );
  });

  it.each([
    [
      "an error",
      [reply, { type: "text", text: "[__COPILOT_ERROR_f7a1__] Rate limited" }],
    ],
    [
      "a note that the turn ended early",
      [
        {
          type: "text",
          text: "[__COPILOT_SYSTEM_e3b0__] The assistant ran into an error and stopped.",
        },
      ],
    ],
    [
      "a call waiting for approval",
      [
        reply,
        {
          ...ranWorkflow,
          output: { type: "approval_required", review_id: "r1" },
        },
      ],
    ],
    [
      "a call with no result",
      [reply, { ...ranWorkflow, state: "input-available", output: undefined }],
    ],
    ...["review_required", "setup_requirements", "need_login"].map((type) => [
      `a result waiting on the user (${type})`,
      [reply, { ...ranWorkflow, output: { type } }],
    ]),
    [
      "a result waiting on the user, then a note about it",
      [
        { ...ranWorkflow, output: { type: "trigger_config_required" } },
        { type: "text", text: "Set up the trigger to continue." },
      ],
    ],
    [
      "a step that failed with nothing after it",
      [reply, { ...ranWorkflow, state: "output-error", errorText: "boom" }],
    ],
    [
      "a pause",
      [reply, { type: "data-turn-paused", data: { reason: "daily" } }],
    ],
    [
      "a reply cut off mid-stream",
      [{ type: "text", text: "Here are your", state: "streaming" }],
    ],
    [
      "thinking cut off mid-stream",
      [{ type: "reasoning", text: "Weighing", state: "streaming" }],
    ],
    ["a frozen turn's note", [reply, { type: "text", text: FROZEN_NOTE }]],
    [
      "an approval card still open after other work",
      [
        { ...ranWorkflow, output: { type: "approval_required" } },
        { ...ranWorkflow, toolCallId: "t2" },
        { type: "text", text: "Waiting on you for the first one." },
      ],
    ],
  ] as [string, unknown[]][])(
    "never says done for a turn that ended in %s",
    (_, parts) => {
      render(
        <TurnStatsBar
          turnMessages={[assistant(parts)]}
          stats={{ durationMs: 12_000 }}
        />,
      );
      const line = screen.getByTestId("turn-end-line");
      expect(line.textContent).not.toMatch(/Done/);
      expect(line.textContent).toMatch(/^Thought for 12s/);
      expect(line.querySelector("svg")).toBeNull();
    },
  );
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

  it.each([
    [
      "a step that failed but was handled",
      [{ ...ranWorkflow, state: "output-error", errorText: "boom" }, reply],
    ],
    [
      "a pause it was resumed from",
      [
        { type: "data-turn-paused", data: { reason: "steps" } },
        ranWorkflow,
        reply,
      ],
    ],
  ] as [string, unknown[]][])(
    "says done for a turn that finished after %s",
    (_, parts) => {
      render(
        <TurnStatsBar
          turnMessages={[assistant(parts)]}
          stats={{ durationMs: 12_000 }}
        />,
      );
      expect(screen.getByTestId("turn-end-line").textContent).toMatch(
        /^Done in 12s/,
      );
    },
  );

  it("says done on the latest turn even with no time or work to show", () => {
    render(<TurnStatsBar turnMessages={[assistant([reply])]} />);
    expect(screen.getByTestId("turn-end-line").textContent).toBe("Done");
  });

  it("does not say done while the latest turn's stream has not settled", () => {
    render(
      <TurnStatsBar
        turnMessages={[assistant([reply])]}
        stats={{ durationMs: 12_000 }}
        isSettled={false}
      />,
    );
    const line = screen.getByTestId("turn-end-line");
    expect(line.textContent).toBe("12s");
    expect(line.querySelector("svg")).toBeNull();
  });

  it("keeps an earlier turn with no time or work to show empty", () => {
    const { container } = render(
      <TurnStatsBar turnMessages={[assistant([reply])]} isLatest={false} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
