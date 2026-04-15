import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { LoadMoreSentinel } from "../ChatMessagesContainer";

vi.mock("use-stick-to-bottom", () => ({
  useStickToBottomContext: () => ({ scrollRef: { current: null } }),
}));

describe("LoadMoreSentinel", () => {
  afterEach(() => {
    cleanup();
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
});
