import { describe, expect, test, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@/tests/integrations/test-utils";
import { LibrarySortMenu } from "../LibrarySortMenu/LibrarySortMenu";

describe("LibrarySortMenu", () => {
  test("renders sort dropdown", () => {
    const setLibrarySort = vi.fn();
    render(<LibrarySortMenu setLibrarySort={setLibrarySort} />);

    expect(screen.getByTestId("sort-by-dropdown")).toBeInTheDocument();
  });

  test("shows 'sort by' label on larger screens", () => {
    const setLibrarySort = vi.fn();
    render(<LibrarySortMenu setLibrarySort={setLibrarySort} />);

    expect(screen.getByText(/sort by/i)).toBeInTheDocument();
  });

  test("shows default placeholder text", () => {
    const setLibrarySort = vi.fn();
    render(<LibrarySortMenu setLibrarySort={setLibrarySort} />);

    expect(screen.getByText(/last modified/i)).toBeInTheDocument();
  });

  test("opens dropdown when clicked", async () => {
    const setLibrarySort = vi.fn();
    render(<LibrarySortMenu setLibrarySort={setLibrarySort} />);

    const trigger = screen.getByRole("combobox");
    fireEvent.click(trigger);

    await waitFor(() => {
      expect(screen.getByText(/creation date/i)).toBeInTheDocument();
    });
  });

  test("shows both sort options in dropdown", async () => {
    const setLibrarySort = vi.fn();
    render(<LibrarySortMenu setLibrarySort={setLibrarySort} />);

    const trigger = screen.getByRole("combobox");
    fireEvent.click(trigger);

    await waitFor(() => {
      expect(screen.getByText(/creation date/i)).toBeInTheDocument();
      expect(
        screen.getAllByText(/last modified/i).length,
      ).toBeGreaterThanOrEqual(1);
    });
  });
});
