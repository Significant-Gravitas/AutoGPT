import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  render as baseRender,
  screen,
} from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { SwapText } from "../SwapText";
import { ToolChain } from "../ToolChain";
import { MIN_RUNNING_MS } from "../useMinimumRunning";

function search(state: string, extras: Record<string, unknown> = {}) {
  return {
    type: "tool-web_search",
    state,
    toolCallId: "call-search",
    input: { query: "hotels" },
    output: state === "output-available" ? { results: [1] } : undefined,
    ...extras,
  } as MessagePart;
}

function chain(parts: MessagePart[], isStreaming: boolean) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()}>
      <ToolChain parts={parts} isStreaming={isStreaming} />
    </CopilotChatActionsProvider>
  );
}

function badgeState(container: HTMLElement) {
  return container.querySelector("[data-state]")?.getAttribute("data-state");
}

function advance(ms: number) {
  act(() => {
    vi.advanceTimersByTime(ms);
  });
}

const RUNNING = 'Searching the web for "hotels"…';

describe("ToolChain timing", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows a running row as running from its first frame", () => {
    const { container } = baseRender(chain([search("input-available")], true));

    expect(badgeState(container)).toBe("spinning");
  });

  it("keeps a call that finished in 80 ms looking like it ran", () => {
    const { container, rerender } = baseRender(
      chain([search("input-available")], true),
    );
    advance(80);
    rerender(chain([search("output-available")], true));

    expect(badgeState(container)).toBe("spinning");
    expect(screen.queryByText('Searched the web for "hotels"')).toBeNull();

    advance(MIN_RUNNING_MS);

    expect(badgeState(container)).toBe("done");
    expect(screen.getByText('Searched the web for "hotels"')).toBeDefined();
  });

  it("shows an error the moment it lands", () => {
    const { container, rerender } = baseRender(
      chain([search("input-available")], true),
    );
    advance(50);
    rerender(chain([search("output-error", { errorText: "Nope" })], true));

    expect(badgeState(container)).toBe("error");
  });

  it("does not hold a call that already ran longer than the minimum", () => {
    const { container, rerender } = baseRender(
      chain([search("input-available")], true),
    );
    advance(MIN_RUNNING_MS + 100);
    rerender(chain([search("output-available")], true));

    expect(badgeState(container)).toBe("done");
  });

  it("gives a newest row that arrives finished its running look", () => {
    const { container } = baseRender(chain([search("output-available")], true));

    expect(badgeState(container)).toBe("spinning");
    advance(MIN_RUNNING_MS);
    expect(badgeState(container)).toBe("done");
  });

  it("draws a reloaded chain as it ended, with no hold and no entrance", () => {
    const { container } = baseRender(
      chain([search("output-available")], false),
    );

    expect(badgeState(container)).toBe("done");
    const header = screen.getByRole("button", { name: /searched the web/i });
    expect(header.parentElement?.style.opacity ?? "").not.toBe("0");
  });

  it("brings a live chain's header in with its first row", () => {
    baseRender(chain([search("input-available")], true));

    const header = screen.getAllByRole("button", { name: RUNNING })[0];
    expect(header.parentElement?.style.opacity).toBe("0");
  });
});

describe("SwapText hold", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("holds each label for its minimum and then lands on the newest", () => {
    const { rerender } = baseRender(<SwapText text="One" minHoldMs={500} />);
    advance(100);
    rerender(<SwapText text="Two" minHoldMs={500} />);
    advance(100);
    rerender(<SwapText text="Three" minHoldMs={500} />);

    expect(screen.getByText("One")).toBeDefined();
    expect(screen.queryByText("Two")).toBeNull();
    expect(screen.queryByText("Three")).toBeNull();

    advance(300);

    expect(screen.getByText("Three")).toBeDefined();
    expect(screen.queryByText("Two")).toBeNull();
  });

  it("shows a change at once when the label has been up long enough", () => {
    const { rerender } = baseRender(<SwapText text="One" minHoldMs={500} />);
    advance(600);
    rerender(<SwapText text="Two" minHoldMs={500} />);

    expect(screen.getByText("Two")).toBeDefined();
  });

  it("shows every change at once without a hold", () => {
    const { rerender } = baseRender(<SwapText text="One" />);
    rerender(<SwapText text="Two" />);

    expect(screen.getByText("Two")).toBeDefined();
  });
});
