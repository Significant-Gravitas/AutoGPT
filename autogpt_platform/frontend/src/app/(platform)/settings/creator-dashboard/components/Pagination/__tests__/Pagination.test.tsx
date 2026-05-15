import { afterEach, describe, expect, it, vi } from "vitest";

import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";

import { Pagination } from "../Pagination";

function makePagination(overrides = {}) {
  return {
    current_page: 2,
    total_pages: 5,
    total_items: 87,
    page_size: 20,
    ...overrides,
  };
}

describe("Pagination", () => {
  afterEach(() => cleanup());

  it("renders nothing when there is only one page", () => {
    const { container } = render(
      <Pagination
        pagination={makePagination({ total_pages: 1 })}
        onPageChange={() => {}}
      />,
    );
    expect(
      container.querySelector('[data-testid="submissions-pagination"]'),
    ).toBeNull();
  });

  it("renders the showing-range copy and both nav buttons", () => {
    render(
      <Pagination pagination={makePagination()} onPageChange={() => {}} />,
    );

    expect(screen.getByText(/showing 21–40 of 87/i)).toBeDefined();
    expect(
      screen.getByRole("button", { name: /previous page/i }),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: /next page/i })).toBeDefined();
  });

  it("clamps the end of the visible range to total_items on the last page", () => {
    render(
      <Pagination
        pagination={makePagination({ current_page: 5, total_pages: 5 })}
        onPageChange={() => {}}
      />,
    );

    expect(screen.getByText(/showing 81–87 of 87/i)).toBeDefined();
  });

  it("disables Previous on page 1 and Next on the last page", () => {
    const { rerender } = render(
      <Pagination
        pagination={makePagination({ current_page: 1 })}
        onPageChange={() => {}}
      />,
    );
    expect(
      (
        screen.getByRole("button", {
          name: /previous page/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);

    rerender(
      <Pagination
        pagination={makePagination({ current_page: 5, total_pages: 5 })}
        onPageChange={() => {}}
      />,
    );
    expect(
      (
        screen.getByRole("button", {
          name: /next page/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  it("invokes onPageChange with current_page ± 1 when buttons are clicked", () => {
    const onPageChange = vi.fn();
    render(
      <Pagination pagination={makePagination()} onPageChange={onPageChange} />,
    );

    fireEvent.click(screen.getByRole("button", { name: /previous page/i }));
    expect(onPageChange).toHaveBeenCalledWith(1);

    fireEvent.click(screen.getByRole("button", { name: /next page/i }));
    expect(onPageChange).toHaveBeenCalledWith(3);
  });

  it("disables both buttons when the disabled prop is true", () => {
    render(
      <Pagination
        pagination={makePagination()}
        onPageChange={() => {}}
        disabled
      />,
    );
    expect(
      (
        screen.getByRole("button", {
          name: /previous page/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
    expect(
      (
        screen.getByRole("button", {
          name: /next page/i,
        }) as HTMLButtonElement
      ).disabled,
    ).toBe(true);
  });

  it("renders every page number when total_pages is small", () => {
    render(
      <Pagination
        pagination={makePagination({ current_page: 2, total_pages: 5 })}
        onPageChange={() => {}}
      />,
    );
    for (const page of [1, 2, 3, 4, 5]) {
      expect(
        screen.getByRole("button", { name: `Go to page ${page}` }),
      ).toBeDefined();
    }
    expect(screen.queryByRole("button", { name: "Go to page 6" })).toBeNull();
  });

  it("slides the visible window so 5 pages render around the current page", () => {
    render(
      <Pagination
        pagination={makePagination({
          current_page: 4,
          total_pages: 10,
          total_items: 200,
        })}
        onPageChange={() => {}}
      />,
    );
    for (const page of [2, 3, 4, 5, 6]) {
      expect(
        screen.getByRole("button", { name: `Go to page ${page}` }),
      ).toBeDefined();
    }
    expect(screen.queryByRole("button", { name: "Go to page 1" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Go to page 7" })).toBeNull();
  });

  it("clamps the window at the end so the last page stays visible", () => {
    render(
      <Pagination
        pagination={makePagination({
          current_page: 10,
          total_pages: 10,
          total_items: 200,
        })}
        onPageChange={() => {}}
      />,
    );
    for (const page of [6, 7, 8, 9, 10]) {
      expect(
        screen.getByRole("button", { name: `Go to page ${page}` }),
      ).toBeDefined();
    }
  });

  it("marks the current page button with aria-current=page", () => {
    render(
      <Pagination
        pagination={makePagination({ current_page: 3, total_pages: 5 })}
        onPageChange={() => {}}
      />,
    );
    const current = screen.getByRole("button", { name: "Go to page 3" });
    expect(current.getAttribute("aria-current")).toBe("page");
    const other = screen.getByRole("button", { name: "Go to page 2" });
    expect(other.getAttribute("aria-current")).toBeNull();
  });

  it("invokes onPageChange with the page number when a numbered button is clicked", () => {
    const onPageChange = vi.fn();
    render(
      <Pagination
        pagination={makePagination({ current_page: 2, total_pages: 5 })}
        onPageChange={onPageChange}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Go to page 5" }));
    expect(onPageChange).toHaveBeenCalledWith(5);
  });
});
