import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { PaginationControls } from "../PaginationControls";

const { push, searchParams } = vi.hoisted(() => ({
  push: vi.fn(),
  searchParams: { current: new URLSearchParams() },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push }),
  usePathname: () => "/admin/marketplace",
  useSearchParams: () => searchParams.current,
}));

describe("PaginationControls", () => {
  beforeEach(() => {
    push.mockClear();
    searchParams.current = new URLSearchParams("status=PENDING&page=2");
  });

  it("shows the current page out of the total", () => {
    render(<PaginationControls currentPage={2} totalPages={5} />);
    expect(screen.getByText("Page 2 of 5")).toBeDefined();
  });

  it("moves to the next and previous page, keeping the other query params", async () => {
    render(<PaginationControls currentPage={2} totalPages={5} />);

    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(push).toHaveBeenLastCalledWith(
      "/admin/marketplace?status=PENDING&page=3",
    );

    await userEvent.click(screen.getByRole("button", { name: "Previous" }));
    expect(push).toHaveBeenLastCalledWith(
      "/admin/marketplace?status=PENDING&page=1",
    );
  });

  it("disables Previous on the first page and Next on the last", () => {
    const { rerender } = render(
      <PaginationControls currentPage={1} totalPages={3} />,
    );
    expect(
      (screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
    expect(
      (screen.getByRole("button", { name: "Next" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);

    rerender(<PaginationControls currentPage={3} totalPages={3} />);
    expect(
      (screen.getByRole("button", { name: "Previous" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false);
    expect(
      (screen.getByRole("button", { name: "Next" }) as HTMLButtonElement)
        .disabled,
    ).toBe(true);
  });

  it("writes the page under a custom query param", async () => {
    searchParams.current = new URLSearchParams();
    render(<PaginationControls currentPage={1} totalPages={2} pathParam="p" />);
    await userEvent.click(screen.getByRole("button", { name: "Next" }));
    expect(push).toHaveBeenCalledWith("/admin/marketplace?p=2");
  });
});
