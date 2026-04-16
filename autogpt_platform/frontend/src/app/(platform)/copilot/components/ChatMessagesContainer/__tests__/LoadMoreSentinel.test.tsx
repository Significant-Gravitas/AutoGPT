import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { LoadMoreSentinel } from "../ChatMessagesContainer";

const mockScrollEl = {
  scrollHeight: 100,
  scrollTop: 0,
  clientHeight: 500,
};

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({ scrollRef: { current: mockScrollEl } }),
}));

type ObserverCallback = (entries: { isIntersecting: boolean }[]) => void;

class MockIntersectionObserver {
  static lastCallback: ObserverCallback | null = null;
  static lastOptions: IntersectionObserverInit | undefined = undefined;
  private callback: ObserverCallback;
  constructor(cb: ObserverCallback, options?: IntersectionObserverInit) {
    this.callback = cb;
    MockIntersectionObserver.lastCallback = cb;
    MockIntersectionObserver.lastOptions = options;
  }
  observe() {}
  disconnect() {}
  unobserve() {}
  takeRecords() {
    return [];
  }
  root = null;
  rootMargin = "";
  thresholds = [];
  fire(entries: { isIntersecting: boolean }[]) {
    this.callback(entries);
  }
}

describe("LoadMoreSentinel", () => {
  beforeEach(() => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    mockScrollEl.clientHeight = 500;
    MockIntersectionObserver.lastCallback = null;
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("renders 'Load older messages' button when hasMore is true and not loading", () => {
    render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={vi.fn()}
      />,
    );
    expect(
      screen.getByRole("button", { name: /load older messages/i }),
    ).toBeDefined();
  });

  it("calls onLoadMore when the button is clicked", () => {
    const onLoadMore = vi.fn();
    render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    fireEvent.click(
      screen.getByRole("button", { name: /load older messages/i }),
    );
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("hides the button and shows a spinner while loading", () => {
    render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={true}
        messageCount={5}
        onLoadMore={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
    expect(screen.getByTestId("load-more-spinner")).toBeDefined();
  });

  it("hides the button when hasMore is false", () => {
    render(
      <LoadMoreSentinel
        hasMore={false}
        isLoading={false}
        messageCount={5}
        onLoadMore={vi.fn()}
      />,
    );
    expect(
      screen.queryByRole("button", { name: /load older messages/i }),
    ).toBeNull();
  });

  it("triggers onLoadMore when the IntersectionObserver fires", () => {
    const onLoadMore = vi.fn();
    render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    expect(MockIntersectionObserver.lastCallback).toBeDefined();
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(1);
  });

  it("ignores observer entries that are not intersecting", () => {
    const onLoadMore = vi.fn();
    render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: false }]);
    expect(onLoadMore).not.toHaveBeenCalled();
  });

  it("restores scroll position after older messages are prepended", () => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 0;
    const onLoadMore = vi.fn();
    const { rerender } = render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    // Auto-fire via observer — this captures the snapshot (prev 100/0).
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    // Simulate DOM growing from prepended older messages.
    mockScrollEl.scrollHeight = 300;
    rerender(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={10}
        onLoadMore={onLoadMore}
      />,
    );
    // scrollTop should be restored to prev + delta = 0 + (300 - 100) = 200.
    expect(mockScrollEl.scrollTop).toBe(200);
  });

  it("does NOT adjust scroll when adjustScroll=false (forward pagination)", () => {
    mockScrollEl.scrollHeight = 100;
    mockScrollEl.scrollTop = 50;
    const onLoadMore = vi.fn();
    const { rerender } = render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
        adjustScroll={false}
      />,
    );
    // Fire observer to capture snapshot.
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    // Simulate DOM growing from appended newer messages (forward load-more).
    mockScrollEl.scrollHeight = 300;
    rerender(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={10}
        onLoadMore={onLoadMore}
        adjustScroll={false}
      />,
    );
    // scrollTop should remain unchanged — no jump for forward pagination.
    expect(mockScrollEl.scrollTop).toBe(50);
  });

  it("ignores same-frame duplicate triggers until isLoading transitions", () => {
    const onLoadMore = vi.fn();
    const { rerender } = render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    // Two observer fires back-to-back — the second must be a no-op while
    // the first load is still pending (isLoading hasn't propagated yet).
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(1);
    // A manual click in the same window is also blocked.
    fireEvent.click(
      screen.getByRole("button", { name: /load older messages/i }),
    );
    expect(onLoadMore).toHaveBeenCalledTimes(1);
    // Simulate parent flipping isLoading on then off — load cycle settled.
    rerender(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={true}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    rerender(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={6}
        onLoadMore={onLoadMore}
      />,
    );
    // Now a fresh trigger should fire again.
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(2);
  });

  function simulateLoadCycle(
    rerender: (ui: React.ReactElement) => void,
    props: {
      hasMore: boolean;
      messageCount: number;
      onLoadMore: () => void;
    },
  ) {
    // Parent pattern: isLoading goes true while fetching, then false with
    // a higher messageCount once new messages land.
    rerender(
      <LoadMoreSentinel
        hasMore={props.hasMore}
        isLoading={true}
        messageCount={props.messageCount - 1}
        onLoadMore={props.onLoadMore}
      />,
    );
    rerender(
      <LoadMoreSentinel
        hasMore={props.hasMore}
        isLoading={false}
        messageCount={props.messageCount}
        onLoadMore={props.onLoadMore}
      />,
    );
  }

  it("resets the auto-fill backoff once the container becomes scrollable via a manual click", () => {
    mockScrollEl.clientHeight = 1000;
    mockScrollEl.scrollHeight = 100;
    const onLoadMore = vi.fn();
    const { rerender } = render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    for (let round = 1; round <= 3; round++) {
      MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
      mockScrollEl.scrollHeight += 50;
      simulateLoadCycle(rerender, {
        hasMore: true,
        messageCount: 5 + round,
        onLoadMore,
      });
    }
    fireEvent.click(
      screen.getByRole("button", { name: /load older messages/i }),
    );
    mockScrollEl.scrollHeight = 2000;
    simulateLoadCycle(rerender, {
      hasMore: true,
      messageCount: 9,
      onLoadMore,
    });
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(5);
  });

  it("stops auto-triggering after 3 non-scrollable rounds but keeps the manual button working", () => {
    mockScrollEl.clientHeight = 1000;
    mockScrollEl.scrollHeight = 100;
    const onLoadMore = vi.fn();
    const { rerender } = render(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={5}
        onLoadMore={onLoadMore}
      />,
    );
    for (let round = 1; round <= 3; round++) {
      MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
      mockScrollEl.scrollHeight += 50;
      simulateLoadCycle(rerender, {
        hasMore: true,
        messageCount: 5 + round,
        onLoadMore,
      });
    }
    expect(onLoadMore).toHaveBeenCalledTimes(3);

    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(3);

    fireEvent.click(
      screen.getByRole("button", { name: /load older messages/i }),
    );
    expect(onLoadMore).toHaveBeenCalledTimes(4);
  });
});
