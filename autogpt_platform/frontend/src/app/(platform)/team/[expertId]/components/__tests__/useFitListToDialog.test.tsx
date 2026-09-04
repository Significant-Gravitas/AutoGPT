import { render } from "@testing-library/react";
import { useRef } from "react";
import { describe, expect, test } from "vitest";
import { useFitListToDialog } from "../useFitListToDialog";

function Harness() {
  const listRef = useRef<HTMLUListElement>(null);
  useFitListToDialog(listRef);
  return (
    <div data-dialog-content>
      <div className="overflow-y-auto" data-testid="body">
        <p>title</p>
        <ul ref={listRef} data-testid="list" />
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
    const { getByTestId, rerender } = render(<Harness />);
    const body = getByTestId("body");
    const list = getByTestId("list");
    mockMetrics(body, list, {
      scrollHeight: 700,
      clientHeight: 500,
      listHeight: 400,
    });

    rerender(<Harness />);

    expect(list.style.maxHeight).toBe("200px");
  });

  test("leaves the list unbounded when nothing overflows", () => {
    const { getByTestId, rerender } = render(<Harness />);
    const body = getByTestId("body");
    const list = getByTestId("list");
    mockMetrics(body, list, {
      scrollHeight: 300,
      clientHeight: 500,
      listHeight: 120,
    });

    rerender(<Harness />);

    expect(list.style.maxHeight).toBe("");
  });

  test("never shrinks the list below a usable minimum", () => {
    const { getByTestId, rerender } = render(<Harness />);
    const body = getByTestId("body");
    const list = getByTestId("list");
    mockMetrics(body, list, {
      scrollHeight: 900,
      clientHeight: 500,
      listHeight: 420,
    });

    rerender(<Harness />);

    expect(list.style.maxHeight).toBe("96px");
  });
});
