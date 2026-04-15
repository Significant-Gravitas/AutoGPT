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

  it("resets the auto-fill backoff once the container becomes scrollable via a manual click", () => {
    // Start non-scrollable: clientHeight > scrollHeight.
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
    // Three auto-triggered loads that leave the container non-scrollable → cap reached.
    for (let round = 1; round <= 3; round++) {
      MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
      mockScrollEl.scrollHeight += 50;
      rerender(
        <LoadMoreSentinel
          hasMore={true}
          isLoading={false}
          messageCount={5 + round}
          onLoadMore={onLoadMore}
        />,
      );
    }
    // Manual button click that makes the container scrollable should reset
    // the counter so auto-fill works again on subsequent observer fires.
    fireEvent.click(
      screen.getByRole("button", { name: /load older messages/i }),
    );
    mockScrollEl.scrollHeight = 2000; // > clientHeight → scrollable
    rerender(
      <LoadMoreSentinel
        hasMore={true}
        isLoading={false}
        messageCount={9}
        onLoadMore={onLoadMore}
      />,
    );
    // Auto-fire should work again after the reset.
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(5);
  });

  it("stops auto-triggering after 3 non-scrollable rounds but keeps the manual button working", () => {
    // clientHeight < scrollHeight would be "scrollable". Here we want
    // non-scrollable: scrollHeight <= clientHeight.
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

    // Simulate three auto-triggered loads, each followed by a
    // messageCount bump (committing the load) while still non-scrollable.
    for (let round = 1; round <= 3; round++) {
      MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
      mockScrollEl.scrollHeight += 50; // still < clientHeight
      rerender(
        <LoadMoreSentinel
          hasMore={true}
          isLoading={false}
          messageCount={5 + round}
          onLoadMore={onLoadMore}
        />,
      );
    }
    expect(onLoadMore).toHaveBeenCalledTimes(3);

    // Fourth observer firing is blocked by the cap.
    MockIntersectionObserver.lastCallback?.([{ isIntersecting: true }]);
    expect(onLoadMore).toHaveBeenCalledTimes(3);

    // Manual button click still works.
    fireEvent.click(
      screen.getByRole("button", { name: /load older messages/i }),
    );
    expect(onLoadMore).toHaveBeenCalledTimes(4);
  });
});
