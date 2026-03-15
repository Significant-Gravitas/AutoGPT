import type { Meta, StoryObj } from "@storybook/nextjs";
import React from "react";
import { InfiniteList } from "./InfiniteList";

const meta: Meta<typeof InfiniteList> = {
  title: "Molecules/InfiniteList",
  component: InfiniteList,
};

export default meta;
type Story = StoryObj<typeof InfiniteList>;

function useMockInfiniteData(total: number, pageSize: number) {
  const [items, setItems] = React.useState<number[]>(
    Array.from({ length: Math.min(pageSize, total) }, (_, i) => i + 1),
  );
  const [isFetchingMore, setIsFetchingMore] = React.useState(false);

  const hasMore = items.length < total;

  function fetchMore() {
    if (!hasMore || isFetchingMore) return;
    setIsFetchingMore(true);
    setTimeout(() => {
      setItems((prev) => {
        const nextStart = prev.length + 1;
        const nextEnd = Math.min(prev.length + pageSize, total);
        const next = Array.from(
          { length: nextEnd - nextStart + 1 },
          (_, i) => nextStart + i,
        );
        return [...prev, ...next];
      });
      setIsFetchingMore(false);
    }, 400);
  }

  return { items, isFetchingMore, hasMore, fetchMore };
}

export const Basic: Story = {
  render: () => {
    const { items, hasMore, isFetchingMore, fetchMore } = useMockInfiniteData(
      40,
      10,
    );

    return (
      <div
        style={{
          height: 320,
          overflow: "auto",
          border: "1px solid #eee",
          padding: 8,
        }}
      >
        <InfiniteList
          items={items}
          hasMore={hasMore}
          isFetchingMore={isFetchingMore}
          onEndReached={fetchMore}
          renderItem={(n) => (
            <div
              style={{
                padding: 8,
                marginBottom: 8,
                background: "#fff",
                border: "1px solid #e5e5e5",
                borderRadius: 8,
              }}
            >
              Item {n}
            </div>
          )}
        />
      </div>
    );
  },
};

export const LongList: Story = {
  render: () => {
    const { items, hasMore, isFetchingMore, fetchMore } = useMockInfiniteData(
      200,
      20,
    );

    return (
      <div
        style={{
          height: 320,
          overflow: "auto",
          border: "1px solid #eee",
          padding: 8,
        }}
      >
        <InfiniteList
          items={items}
          hasMore={hasMore}
          isFetchingMore={isFetchingMore}
          onEndReached={fetchMore}
          renderItem={(n) => (
            <div
              style={{
                padding: 8,
                marginBottom: 8,
                background: "#fff",
                border: "1px solid #e5e5e5",
                borderRadius: 8,
              }}
            >
              Row {n}
            </div>
          )}
        />
      </div>
    );
  },
};

export const WithLoadingIndicator: Story = {
  render: () => {
    const { items, hasMore, isFetchingMore, fetchMore } = useMockInfiniteData(
      100,
      10,
    );

    return (
      <div
        style={{
          height: 320,
          overflow: "auto",
          border: "1px solid #eee",
          padding: 8,
        }}
      >
        <InfiniteList
          items={items}
          hasMore={hasMore}
          isFetchingMore={isFetchingMore}
          onEndReached={fetchMore}
          renderItem={(n) => (
            <div
              style={{
                padding: 8,
                marginBottom: 8,
                background: "#fff",
                border: "1px solid #e5e5e5",
                borderRadius: 8,
              }}
            >
              #{n}
            </div>
          )}
        />
        {isFetchingMore && (
          <div style={{ padding: 8, color: "#666" }}>Loading more…</div>
        )}
      </div>
    );
  },
};
