import { describe, expect, test, vi } from "vitest";
import { render, screen, fireEvent, waitFor } from "@/tests/integrations/test-utils";
import { LibrarySearchBar } from "../LibrarySearchBar/LibrarySearchBar";

describe("LibrarySearchBar", () => {
  test("renders search input", () => {
    const setSearchTerm = vi.fn();
    render(<LibrarySearchBar setSearchTerm={setSearchTerm} />);

    expect(screen.getByPlaceholderText(/search agents/i)).toBeInTheDocument();
  });

  test("renders search icon", () => {
    const setSearchTerm = vi.fn();
    const { container } = render(
      <LibrarySearchBar setSearchTerm={setSearchTerm} />,
    );

    // Check for the magnifying glass icon (SVG element)
    const searchIcon = container.querySelector("svg");
    expect(searchIcon).toBeInTheDocument();
  });

  test("calls setSearchTerm on input change", async () => {
    const setSearchTerm = vi.fn();
    render(<LibrarySearchBar setSearchTerm={setSearchTerm} />);

    const input = screen.getByPlaceholderText(/search agents/i);
    fireEvent.change(input, { target: { value: "test query" } });

    // The search bar uses debouncing, so we need to wait
    await waitFor(
      () => {
        expect(setSearchTerm).toHaveBeenCalled();
      },
      { timeout: 1000 },
    );
  });

  test("has correct test id", () => {
    const setSearchTerm = vi.fn();
    render(<LibrarySearchBar setSearchTerm={setSearchTerm} />);

    expect(screen.getByTestId("search-bar")).toBeInTheDocument();
  });

  test("input has correct test id", () => {
    const setSearchTerm = vi.fn();
    render(<LibrarySearchBar setSearchTerm={setSearchTerm} />);

    expect(screen.getByTestId("library-textbox")).toBeInTheDocument();
  });
});
