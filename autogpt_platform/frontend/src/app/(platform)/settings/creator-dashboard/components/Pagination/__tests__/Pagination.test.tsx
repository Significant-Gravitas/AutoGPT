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
});
