import { fireEvent, render, screen } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { getPageItems } from "./helpers";
import { Pagination } from "./Pagination";

describe("getPageItems", () => {
  it("lists every page when they all fit", () => {
    expect(getPageItems(1, 5)).toEqual([1, 2, 3, 4, 5]);
    expect(getPageItems(4, 7)).toEqual([1, 2, 3, 4, 5, 6, 7]);
  });

  it("puts an ellipsis on the far side near either end", () => {
    expect(getPageItems(1, 10)).toEqual([1, 2, 3, 4, 5, "ellipsis-end", 10]);
    expect(getPageItems(10, 10)).toEqual([1, "ellipsis-start", 6, 7, 8, 9, 10]);
  });

  it("puts an ellipsis on both sides in the middle", () => {
    expect(getPageItems(5, 10)).toEqual([
      1,
      "ellipsis-start",
      4,
      5,
      6,
      "ellipsis-end",
      10,
    ]);
    expect(getPageItems(20, 40, 2)).toEqual([
      1,
      "ellipsis-start",
      18,
      19,
      20,
      21,
      22,
      "ellipsis-end",
      40,
    ]);
  });
});

describe("Pagination", () => {
  it("marks the current page and changes page on click", () => {
    const onPageChange = vi.fn();
    render(<Pagination page={5} pageCount={10} onPageChange={onPageChange} />);

    expect(
      screen.getByRole("navigation", { name: "Pagination" }),
    ).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "Page 5" })
        .getAttribute("aria-current"),
    ).toBe("page");
    expect(
      screen
        .getByRole("button", { name: "Page 4" })
        .getAttribute("aria-current"),
    ).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Page 6" }));
    fireEvent.click(screen.getByRole("button", { name: /Previous/ }));
    fireEvent.click(screen.getByRole("button", { name: /Next/ }));

    expect(onPageChange.mock.calls).toEqual([[6], [4], [6]]);
  });

  it("does not report the current page again", () => {
    const onPageChange = vi.fn();
    render(<Pagination page={2} pageCount={3} onPageChange={onPageChange} />);

    fireEvent.click(screen.getByRole("button", { name: "Page 2" }));

    expect(onPageChange).not.toHaveBeenCalled();
  });

  it("disables previous on the first page and next on the last", () => {
    const { rerender } = render(
      <Pagination page={1} pageCount={3} onPageChange={vi.fn()} />,
    );
    expect(
      (screen.getByRole("button", { name: /Previous/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: /Next/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);

    rerender(<Pagination page={3} pageCount={3} onPageChange={vi.fn()} />);
    expect(
      (screen.getByRole("button", { name: /Next/ }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("disables every button when disabled and forwards its ref", () => {
    const ref = createRef<HTMLElement>();
    render(
      <Pagination
        ref={ref}
        page={2}
        pageCount={3}
        onPageChange={vi.fn()}
        disabled
      />,
    );

    for (const button of screen.getAllByRole("button")) {
      expect((button as HTMLButtonElement).disabled).toBe(true);
    }
    expect(ref.current?.tagName).toBe("NAV");
  });
});
