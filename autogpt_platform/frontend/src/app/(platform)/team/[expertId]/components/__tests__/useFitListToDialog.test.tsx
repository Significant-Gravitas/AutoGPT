import { fireEvent, render } from "@testing-library/react";
import { describe, expect, test, vi } from "vitest";
import { useFitListToDialog } from "../useFitListToDialog";

function Harness() {
  const { attachList } = useFitListToDialog<HTMLUListElement>();
  return (
    <div data-dialog-content>
      <div className="overflow-y-auto" data-testid="body">
        <p>title</p>
        <ul ref={attachList} data-testid="list" />
      </div>
    </div>
  );
}

function mockMetrics(
  body: HTMLElement,
  list: HTMLElement,
  metrics: { scrollHeight: number; clientHeight: number; listHeight: number },
) {
  Object.defineProperty(body, "scrollHeight", {
    configurable: true,
    get: () => metrics.scrollHeight,
  });
  Object.defineProperty(body, "clientHeight", {
    configurable: true,
    get: () => metrics.clientHeight,
  });
  Object.defineProperty(list, "offsetHeight", {
    configurable: true,
    get: () => metrics.listHeight,
  });
}

describe("useFitListToDialog", () => {
  test("caps the list by exactly the body's overflow", () => {
    const { getByTestId } = render(<Harness />);
    const body = getByTestId("body");
    const list = getByTestId("list");
    mockMetrics(body, list, {
      scrollHeight: 700,
      clientHeight: 500,
      listHeight: 400,
    });

    fireEvent(window, new Event("resize"));

    expect(list.style.maxHeight).toBe("200px");
  });

  test("leaves the list unbounded when nothing overflows", () => {
    const { getByTestId } = render(<Harness />);
    const body = getByTestId("body");
    const list = getByTestId("list");
    mockMetrics(body, list, {
      scrollHeight: 300,
      clientHeight: 500,
      listHeight: 120,
    });

    fireEvent(window, new Event("resize"));

    expect(list.style.maxHeight).toBe("");
  });

  test("never shrinks the list below a usable minimum", () => {
    const { getByTestId } = render(<Harness />);
    const body = getByTestId("body");
    const list = getByTestId("list");
    mockMetrics(body, list, {
      scrollHeight: 900,
      clientHeight: 500,
      listHeight: 420,
    });

    fireEvent(window, new Event("resize"));

    expect(list.style.maxHeight).toBe("96px");
  });
  test("keeps observers stable on rerenders and cleans them up on unmount", () => {
    const add = vi.spyOn(window, "addEventListener");
    const remove = vi.spyOn(window, "removeEventListener");
    const { rerender, unmount } = render(<Harness />);
    const subscriptions = add.mock.calls.filter(
      ([event]) => event === "resize",
    ).length;
    rerender(<Harness />);
    rerender(<Harness />);
    expect(add.mock.calls.filter(([event]) => event === "resize")).toHaveLength(
      subscriptions,
    );
    unmount();
    expect(remove.mock.calls.some(([event]) => event === "resize")).toBe(true);
    add.mockRestore();
    remove.mockRestore();
  });
});
