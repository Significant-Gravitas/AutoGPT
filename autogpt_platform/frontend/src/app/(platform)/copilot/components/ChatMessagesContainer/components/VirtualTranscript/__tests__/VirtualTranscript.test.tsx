import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { VirtualTranscript } from "../VirtualTranscript";
import type { TranscriptRow } from "../helpers";
import type { JumpToMessage } from "../useVirtualTranscript";

type ChatMessage = UIMessage<unknown, UIDataTypes, UITools>;

// Every row measures what it is estimated at (a short text row), so the
// window's arithmetic is exact: 96 px rows with 24 px gaps.
const ROW_HEIGHT = 96;
const ROW_STRIDE = 120;
const VIEWPORT = 800;

const stick = vi.hoisted(() => ({
  scroller: null as HTMLDivElement | null,
  state: { isAtBottom: true },
}));

// A ref object, as StickToBottom's is: read at the moment it is used.
const scrollRef = {
  get current() {
    return stick.scroller;
  },
};

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({ scrollRef, state: stick.state }),
}));

function makeScroller() {
  const scroller = document.createElement("div");
  Object.defineProperty(scroller, "clientHeight", {
    configurable: true,
    value: VIEWPORT,
  });
  Object.defineProperty(scroller, "offsetHeight", { value: VIEWPORT });
  Object.defineProperty(scroller, "offsetWidth", { value: 600 });
  scroller.getBoundingClientRect = () => ({ top: 0 }) as DOMRect;
  Object.defineProperty(scroller, "scrollHeight", { value: 1_000_000 });
  // The browser fires the scroll event later, never inside the call.
  scroller.scrollTo = ((options: ScrollToOptions) => {
    scroller.scrollTop = options.top ?? 0;
  }) as typeof scroller.scrollTo;
  document.body.appendChild(scroller);
  return scroller;
}

function message(index: number, role?: "user" | "assistant"): ChatMessage {
  return {
    id: `m${index}`,
    role: role ?? (index % 2 === 0 ? "user" : "assistant"),
    parts: [{ type: "text", text: `Message ${index}` }],
  } as ChatMessage;
}

function toRows(messages: ChatMessage[]): TranscriptRow[] {
  return messages.map((m, rowIndex) => ({ message: m, rowIndex }));
}

function Counter({ id }: { id: string }) {
  const [count, setCount] = useState(0);
  return (
    <button type="button" onClick={() => setCount((c) => c + 1)}>
      {id} clicked {count}
    </button>
  );
}

let jump: JumpToMessage | null = null;

function Transcript({ messages }: { messages: ChatMessage[] }) {
  const jumpRef = useRef<JumpToMessage | null>(null);
  jump = (id) => jumpRef.current?.(id);
  const lastUser = messages.findLast((m) => m.role === "user");
  return (
    <VirtualTranscript
      rows={toRows(messages)}
      renderRow={({ message: m }) => (
        <div data-message-id={m.id}>
          <span>{(m.parts[0] as { text: string }).text}</span>
          <Counter id={m.id} />
        </div>
      )}
      lastUserMessageID={lastUser?.id ?? null}
      compact={false}
      jumpRef={jumpRef}
    />
  );
}

function mountedIDs() {
  return Array.from(document.querySelectorAll("[data-message-id]")).map(
    (node) => node.getAttribute("data-message-id"),
  );
}

function scrollTo(top: number) {
  act(() => {
    stick.scroller!.scrollTop = top;
    stick.scroller!.dispatchEvent(new Event("scroll"));
  });
}

function bottomOffset(count: number) {
  return count * ROW_STRIDE - 24 - VIEWPORT;
}

function renderTranscript(messages: ChatMessage[]) {
  return render(<Transcript messages={messages} />, {
    container: stick.scroller!,
  });
}

const offsetHeight = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  "offsetHeight",
);
const getBoundingClientRect = HTMLElement.prototype.getBoundingClientRect;

describe("VirtualTranscript", () => {
  beforeEach(() => {
    stick.scroller = makeScroller();
    stick.state.isAtBottom = true;
    Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
      configurable: true,
      get(this: HTMLElement) {
        return this.dataset.index !== undefined ? ROW_HEIGHT : 0;
      },
    });
    // Everything inside the scroller sits at the top of its content.
    HTMLElement.prototype.getBoundingClientRect = function () {
      return { top: -(stick.scroller?.scrollTop ?? 0) } as DOMRect;
    };
  });

  afterEach(() => {
    cleanup();
    stick.scroller?.remove();
    HTMLElement.prototype.getBoundingClientRect = getBoundingClientRect;
    if (offsetHeight) {
      Object.defineProperty(
        HTMLElement.prototype,
        "offsetHeight",
        offsetHeight,
      );
    }
  });

  it("mounts only the rows near the viewport of a 500-message thread", () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    renderTranscript(messages);
    scrollTo(bottomOffset(500));

    const ids = mountedIDs();
    expect(ids).toContain("m499");
    expect(ids).not.toContain("m0");
    // About 7 rows fit the viewport, plus 6 overscan rows either side.
    expect(ids.length).toBeLessThanOrEqual(7 + 2 * 6 + 1);
  });

  it("keeps the list as tall as every row would be", () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    renderTranscript(messages);
    scrollTo(bottomOffset(500));
    const list = document.querySelector("[data-index]")!
      .parentElement as HTMLElement;
    const top = parseFloat(list.style.paddingTop || "0");
    const mounted = mountedIDs().length;
    // Rows above the window collapse into padding: whole rows plus gaps.
    expect(top).toBe((500 - mounted) * ROW_STRIDE);
  });

  it("windows the oldest rows when the user scrolls to the top, keeping the newest turn", () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    renderTranscript(messages);
    stick.state.isAtBottom = false;
    scrollTo(0);

    const ids = mountedIDs();
    expect(ids).toContain("m0");
    // The newest user message and the reply below it stay mounted.
    expect(ids).toContain("m498");
    expect(ids).toContain("m499");
    expect(ids).not.toContain("m250");
  });

  it("keeps the newest user message mounted when its turn runs past eight rows", () => {
    // The newest user message is row 480; nineteen assistant rows follow it.
    const messages = Array.from({ length: 500 }, (_, i) =>
      message(i, i <= 480 ? undefined : "assistant"),
    );
    renderTranscript(messages);
    stick.state.isAtBottom = false;
    scrollTo(0);

    const ids = mountedIDs();
    // TailSpacer measures the turn from this row.
    expect(ids).toContain("m480");
    expect(ids).toContain("m499");
    expect(ids).not.toContain("m485");
  });

  it("mounts a message appended while pinned to the bottom", () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    const { rerender } = renderTranscript(messages);
    scrollTo(bottomOffset(500));

    const next = [...messages, message(500, "user")];
    rerender(<Transcript messages={next} />);
    expect(mountedIDs()).toContain("m500");
  });

  it("does not move the scroll position when a message is appended while scrolled up", () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    const { rerender } = renderTranscript(messages);
    stick.state.isAtBottom = false;
    scrollTo(10 * ROW_STRIDE);

    rerender(<Transcript messages={[...messages, message(500, "user")]} />);
    expect(stick.scroller!.scrollTop).toBe(10 * ROW_STRIDE);
    expect(mountedIDs()).toContain("m10");
  });

  it("keeps the rows on screen mounted when older history is prepended", () => {
    const messages = Array.from({ length: 100 }, (_, i) => message(i + 50));
    const { rerender } = renderTranscript(messages);
    stick.state.isAtBottom = false;
    scrollTo(0);
    act(() => screen.getByRole("button", { name: "m50 clicked 0" }).click());

    const older = Array.from({ length: 50 }, (_, i) => message(i));
    // LoadMoreSentinel moves the scroll position down by what arrived.
    rerender(<Transcript messages={[...older, ...messages]} />);
    scrollTo(50 * ROW_STRIDE);

    expect(screen.getByRole("button", { name: "m50 clicked 1" })).toBeDefined();
  });

  it("scrolls an unmounted message into the window by id", async () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    renderTranscript(messages);
    scrollTo(bottomOffset(500));
    expect(mountedIDs()).not.toContain("m100");

    stick.state.isAtBottom = false;
    act(() => jump?.("m100"));
    act(() => {
      stick.scroller!.dispatchEvent(new Event("scroll"));
    });
    // Let the virtualizer's settle-and-verify frames run out.
    await act(() => new Promise((resolve) => setTimeout(resolve, 100)));

    expect(mountedIDs()).toContain("m100");
    expect(stick.scroller!.scrollTop).toBe(100 * ROW_STRIDE - 352);
  });

  it("skips the entrance animation for rows scrolling back into the window", () => {
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    renderTranscript(messages);
    scrollTo(bottomOffset(500));
    stick.state.isAtBottom = false;
    scrollTo(0);

    const returning = document
      .querySelector('[data-message-id="m0"]')
      ?.closest("[data-index]");
    expect(returning?.className).toContain("[&>*]:animate-none");
  });

  describe("keeps the rows on screen still", () => {
    // Row 470 is taller than its guess: it measures 200 px, 104 px over.
    const TALL_INDEX = 470;
    const TALL_HEIGHT = 200;

    // A small flow layout: each row sits below the list's top padding and
    // the rows before it, 24 px apart.
    function layoutTop(node: HTMLElement) {
      const list = node.parentElement!;
      let top = parseFloat(list.style.paddingTop || "0");
      for (const sibling of Array.from(list.children) as HTMLElement[]) {
        if (sibling === node) break;
        top += (sibling.offsetHeight || parseFloat(sibling.style.height)) + 24;
      }
      return top - stick.scroller!.scrollTop;
    }

    beforeEach(() => {
      Object.defineProperty(HTMLElement.prototype, "offsetHeight", {
        configurable: true,
        get(this: HTMLElement) {
          if (this.dataset.index === undefined) return 0;
          return Number(this.dataset.index) === TALL_INDEX
            ? TALL_HEIGHT
            : ROW_HEIGHT;
        },
      });
      HTMLElement.prototype.getBoundingClientRect = function () {
        if (this.dataset.index === undefined) {
          return { top: -(stick.scroller?.scrollTop ?? 0) } as DOMRect;
        }
        const top = layoutTop(this);
        return { top, bottom: top + this.offsetHeight } as DOMRect;
      };
    });

    it("moves the scroll position by the error of a row that enters above the viewport", () => {
      const messages = Array.from({ length: 500 }, (_, i) => message(i));
      renderTranscript(messages);
      stick.state.isAtBottom = false;
      // Row 478 at the top of the viewport: the window starts six rows
      // above, at 472.
      scrollTo(478 * ROW_STRIDE);
      expect(mountedIDs()).not.toContain(`m${TALL_INDEX}`);

      scrollTo(476 * ROW_STRIDE);

      expect(mountedIDs()).toContain(`m${TALL_INDEX}`);
      expect(stick.scroller!.scrollTop).toBe(
        476 * ROW_STRIDE + (TALL_HEIGHT - ROW_HEIGHT),
      );
    });

    it("leaves the scroll position to StickToBottom while pinned", () => {
      const messages = Array.from({ length: 500 }, (_, i) => message(i));
      renderTranscript(messages);
      scrollTo(478 * ROW_STRIDE);

      scrollTo(476 * ROW_STRIDE);

      expect(mountedIDs()).toContain(`m${TALL_INDEX}`);
      expect(stick.scroller!.scrollTop).toBe(476 * ROW_STRIDE);
    });

    it("puts the row being read back in place when older history arrives", () => {
      const messages = Array.from({ length: 100 }, (_, i) => message(i + 50));
      const { rerender } = renderTranscript(messages);
      stick.state.isAtBottom = false;
      scrollTo(0);

      const older = Array.from({ length: 50 }, (_, i) => message(i));
      // The page also merges ten rows further down, so the total height
      // grows by less than the height that arrived above.
      const merged = messages.filter((_, i) => i < 70 || i >= 80);
      rerender(<Transcript messages={[...older, ...merged]} />);

      expect(mountedIDs()).toContain("m50");
      expect(stick.scroller!.scrollTop).toBe(50 * ROW_STRIDE);
    });

    it("does not move for a row that enters below the viewport", () => {
      const messages = Array.from({ length: 500 }, (_, i) => message(i));
      renderTranscript(messages);
      stick.state.isAtBottom = false;
      scrollTo(457 * ROW_STRIDE);
      expect(mountedIDs()).not.toContain(`m${TALL_INDEX}`);

      scrollTo(458 * ROW_STRIDE);

      expect(mountedIDs()).toContain(`m${TALL_INDEX}`);
      expect(stick.scroller!.scrollTop).toBe(458 * ROW_STRIDE);
    });
  });

  it("windows rows when it mounts in the same commit as the scroll element", () => {
    const scroller = stick.scroller!;
    stick.scroller = null;
    // React attaches an ancestor's ref after its children's layout effects,
    // as when a session switch remounts StickToBottom and the transcript.
    function AttachScroller({ children }: { children: ReactNode }) {
      useLayoutEffect(() => {
        stick.scroller = scroller;
      }, []);
      return children;
    }
    const renderRow = vi.fn(({ message: m }: TranscriptRow) => (
      <div data-message-id={m.id} />
    ));
    const messages = Array.from({ length: 500 }, (_, i) => message(i));
    render(
      <AttachScroller>
        <VirtualTranscript
          rows={toRows(messages)}
          renderRow={renderRow}
          lastUserMessageID="m498"
          compact={false}
        />
      </AttachScroller>,
      { container: scroller },
    );

    expect(mountedIDs().length).toBeLessThan(30);
    // No pass ever rendered the whole thread.
    expect(renderRow.mock.calls.length).toBeLessThan(150);
  });

  it("renders every row when the scroll element has no height", () => {
    Object.defineProperty(stick.scroller, "clientHeight", { value: 0 });
    const messages = Array.from({ length: 60 }, (_, i) => message(i));
    renderTranscript(messages);
    expect(mountedIDs()).toHaveLength(60);
  });

  it("leaves no update behind when it unmounts mid-scroll", () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      renderTranscript(Array.from({ length: 500 }, (_, i) => message(i)));
      scrollTo(bottomOffset(500));
      cleanup();
      // Vitest deletes `window` when it tears the environment down, and an
      // update landing after that fails the run.
      vi.stubGlobal("window", undefined);
      vi.advanceTimersByTime(1_000);
    } finally {
      vi.unstubAllGlobals();
      vi.useRealTimers();
    }
  });
});
