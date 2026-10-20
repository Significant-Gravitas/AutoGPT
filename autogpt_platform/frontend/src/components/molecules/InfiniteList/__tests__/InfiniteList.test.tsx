import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it } from "vitest";
import { InfiniteList } from "../InfiniteList";

interface Item {
  id: string;
}

function Row({ item }: { item: Item }) {
  const [note, setNote] = useState("");
  return (
    <input
      aria-label={item.id}
      value={note}
      onChange={(event) => setNote(event.target.value)}
    />
  );
}

function List({ items, keyed }: { items: Item[]; keyed: boolean }) {
  return (
    <InfiniteList
      items={items}
      hasMore={false}
      onEndReached={() => undefined}
      getKey={keyed ? (item) => item.id : undefined}
      renderItem={(item) => <Row item={item} />}
    />
  );
}

describe("InfiniteList", () => {
  it("keeps each row's state with its item when an item is prepended", async () => {
    const { rerender } = render(<List keyed items={[{ id: "b" }]} />);
    const rowB = screen.getByLabelText("b");
    await userEvent.type(rowB, "kept");

    rerender(<List keyed items={[{ id: "a" }, { id: "b" }]} />);

    expect(screen.getByLabelText("b")).toBe(rowB);
    expect(screen.getByLabelText<HTMLInputElement>("b").value).toBe("kept");
    expect(screen.getByLabelText<HTMLInputElement>("a").value).toBe("");
  });

  it("keys rows by position when no getKey is given", async () => {
    const { rerender } = render(<List keyed={false} items={[{ id: "b" }]} />);
    await userEvent.type(screen.getByLabelText("b"), "moved");

    rerender(<List keyed={false} items={[{ id: "a" }, { id: "b" }]} />);

    expect(screen.getByLabelText<HTMLInputElement>("a").value).toBe("moved");
  });
});
