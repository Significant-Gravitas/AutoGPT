import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it } from "vitest";
import { useTokenDevtoolStore } from "../../../tokenDevtool/store";
import type {
  ContextBreakdown,
  TokenTurn,
} from "../../../tokenDevtool/tokenMath";
import { MODEL_CONTEXT_WINDOW } from "../../../tokenDevtool/tokenMath";
import { windowPercent } from "../helpers";
import { TokenDevtoolBadge } from "../TokenDevtoolBadge";

const SESSION = "session-1";

// BASE_CONTEXT_ESTIMATE (65k) + 5k + 3k + 2k = 75k.
const BREAKDOWN: ContextBreakdown = {
  userTokens: 5000,
  assistantTokens: 3000,
  toolTokens: 2000,
};

function turn(overrides: Partial<TokenTurn> = {}): TokenTurn {
  return {
    promptTokens: 0,
    completionTokens: 0,
    cacheReadTokens: 0,
    cacheCreationTokens: 0,
    compacted: false,
    at: 1,
    ...overrides,
  };
}

function seed({
  turns,
  breakdown,
  sessionId = SESSION,
}: {
  turns?: TokenTurn[];
  breakdown?: ContextBreakdown;
  sessionId?: string;
}) {
  reset();
  useTokenDevtoolStore.setState({
    breakdownBySession: breakdown ? { [sessionId]: breakdown } : {},
  });
  // Route turns through record() so the derived live-context sum and the
  // sticky compacted flag stay consistent with production.
  turns?.forEach((t) => useTokenDevtoolStore.getState().record(sessionId, t));
}

function reset() {
  useTokenDevtoolStore.setState({
    turnsBySession: {},
    breakdownBySession: {},
    liveContextBySession: {},
    compactedBySession: {},
    sessionOrder: [],
  });
}

function getTrigger() {
  return screen.getByRole("button", { name: /Token devtool/ });
}

async function openPopover() {
  await userEvent.click(getTrigger());
  return screen.findByText("Context window");
}

afterEach(reset);

describe("TokenDevtoolBadge trigger", () => {
  it("shows the em-dash state when the session has no data", () => {
    seed({ turns: [turn({ cacheCreationTokens: 40000 })], sessionId: "other" });

    render(<TokenDevtoolBadge sessionId={SESSION} />);

    expect(getTrigger().textContent).toContain("ctx —");
    expect(getTrigger().textContent).not.toContain("~");
  });

  it("shows the seeded history estimate before any live turn", () => {
    seed({ breakdown: BREAKDOWN });

    render(<TokenDevtoolBadge sessionId={SESSION} />);

    expect(getTrigger().textContent).toContain("ctx ~75k");
  });

  it("keeps the seeded estimate while the live cache-write sum is below it", () => {
    seed({
      breakdown: BREAKDOWN,
      turns: [turn({ cacheCreationTokens: 20000 })],
    });

    render(<TokenDevtoolBadge sessionId={SESSION} />);

    expect(getTrigger().textContent).toContain("ctx ~75k");
  });

  it("switches to the live cache-write sum once it exceeds the seed", () => {
    seed({
      breakdown: BREAKDOWN,
      turns: [
        turn({ cacheCreationTokens: 20000 }),
        turn({ cacheCreationTokens: 90000 }),
      ],
    });

    render(<TokenDevtoolBadge sessionId={SESSION} />);

    expect(getTrigger().textContent).toContain("ctx ~110k");
  });

  it("drops the seed and restarts the sum after a compaction turn", () => {
    seed({
      breakdown: BREAKDOWN,
      turns: [
        turn({ cacheCreationTokens: 90000 }),
        turn({ cacheCreationTokens: 40000, compacted: true }),
      ],
    });

    render(<TokenDevtoolBadge sessionId={SESSION} />);

    expect(getTrigger().textContent).toContain("ctx ~40k");
  });
});

describe("TokenDevtoolBadge popover", () => {
  it("reveals the context window readout against the model window", async () => {
    seed({ breakdown: BREAKDOWN });

    render(<TokenDevtoolBadge sessionId={SESSION} />);
    expect(screen.queryByText("Context window")).toBeNull();

    await openPopover();

    expect(screen.getByText(/~75k \/ 200k/)).toBeDefined();
    expect(screen.getByText("summarizes ~100k")).toBeDefined();
    // The threshold caveat is visible text, not a title on a 1px div.
    expect(
      screen.getByText(/assumes a 200k window; the backend threshold/),
    ).toBeDefined();
  });

  it("shows an em-dash readout when the session has no data", async () => {
    render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();

    expect(screen.getByText(/^— \/ 200k$/)).toBeDefined();
  });

  it("splits the seeded estimate into labelled breakdown rows", async () => {
    seed({ breakdown: BREAKDOWN });

    render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();

    expect(screen.getByText("system + tools + skills")).toBeDefined();
    expect(screen.getByText("fixed est.")).toBeDefined();
    expect(screen.getByText("~65k")).toBeDefined();
    expect(screen.getByText("your messages")).toBeDefined();
    expect(screen.getByText("~5k")).toBeDefined();
    expect(screen.getByText("assistant replies")).toBeDefined();
    expect(screen.getByText("~3k")).toBeDefined();
    expect(screen.getByText("tool calls + results")).toBeDefined();
    expect(screen.getByText("~2k")).toBeDefined();
  });

  it("omits the breakdown section until the history estimate exists", async () => {
    seed({ turns: [turn({ cacheCreationTokens: 20000 })] });

    render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();

    expect(screen.queryByText("your messages")).toBeNull();
    expect(screen.queryByText("system + tools + skills")).toBeNull();
  });

  it("hints that live data starts next message when no turns exist", async () => {
    seed({ breakdown: BREAKDOWN });

    render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();

    expect(
      screen.getByText("Live per-turn data starts with your next message."),
    ).toBeDefined();
  });

  it("lists per-turn input, output, and cache-write usage", async () => {
    seed({
      turns: [
        turn({
          promptTokens: 1200,
          completionTokens: 300,
          cacheReadTokens: 40000,
          cacheCreationTokens: 2000,
        }),
        turn({
          promptTokens: 500,
          completionTokens: 120,
          cacheReadTokens: 60000,
          cacheCreationTokens: 4500,
          at: 2,
        }),
      ],
    });

    render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();

    expect(
      screen.queryByText("Live per-turn data starts with your next message."),
    ).toBeNull();
    expect(screen.getByText("#1")).toBeDefined();
    expect(screen.getByText("in 43.2k")).toBeDefined();
    expect(screen.getByText("out 300")).toBeDefined();
    expect(screen.getByText("w 2k")).toBeDefined();
    expect(screen.getByText("#2")).toBeDefined();
    expect(screen.getByText("in 65k")).toBeDefined();
    expect(screen.getByText("out 120")).toBeDefined();
    expect(screen.getByText("w 4.5k")).toBeDefined();
  });

  it("marks a compacted turn with the ⟲ glyph and only that turn", async () => {
    seed({
      turns: [
        turn({ cacheCreationTokens: 90000 }),
        turn({ cacheCreationTokens: 40000, compacted: true, at: 2 }),
      ],
    });

    render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();

    const markers = screen.getAllByText("transcript summarized this turn");
    expect(markers).toHaveLength(1);
    expect(markers[0].parentElement?.textContent).toContain("⟲");
    expect(screen.getByText("#2")).toBeDefined();
  });
});

describe("TokenDevtoolBadge autocompact threshold", () => {
  // The amber fill is the tool's primary "you are about to be summarized"
  // signal, so both sides of the threshold are pinned.
  async function fillClassesAtContext(userTokens: number) {
    seed({ breakdown: { userTokens, assistantTokens: 0, toolTokens: 0 } });
    const { container } = render(<TokenDevtoolBadge sessionId={SESSION} />);
    await openPopover();
    return Array.from(container.ownerDocument.querySelectorAll("div, span"))
      .map((node) => node.className)
      .filter((name) => typeof name === "string");
  }

  it("turns the bar amber at or above the autocompact trigger", async () => {
    // 65k base + 40k = 105k, over the 100k trigger.
    const classes = await fillClassesAtContext(40_000);
    expect(classes.some((name) => name.includes("bg-amber-400"))).toBe(true);
  });

  it("keeps the bar neutral below the autocompact trigger", async () => {
    // 65k base + 5k = 70k, under the trigger.
    const classes = await fillClassesAtContext(5_000);
    expect(classes.some((name) => name.includes("bg-zinc-800"))).toBe(true);
  });
});

describe("windowPercent", () => {
  it("clamps a context beyond the model window to 100", () => {
    expect(windowPercent(MODEL_CONTEXT_WINDOW * 2)).toBe(100);
    expect(windowPercent(MODEL_CONTEXT_WINDOW / 2)).toBe(50);
  });
});
