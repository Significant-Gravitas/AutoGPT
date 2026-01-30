import { describe, expect, test } from "vitest";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";

const defaultProps = {
  searchTerm: "test-search",
  sort: undefined as undefined,
};

describe("MainSearchResultPage - Rendering", () => {
  test("renders search results header with search term", async () => {
    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByText("Results for:")).toBeInTheDocument();
    });
    expect(screen.getByText("test-search")).toBeInTheDocument();
  });

  test("renders search bar", async () => {
    render(<MainSearchResultPage {...defaultProps} />);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
    });
  });
});
