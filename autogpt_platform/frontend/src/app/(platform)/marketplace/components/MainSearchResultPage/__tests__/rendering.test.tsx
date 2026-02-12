import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainSearchResultPage } from "../MainSearchResultPage";

const defaultProps = {
  searchTerm: "test-search",
  sort: undefined,
};

describe("MainSearchResultPage - Rendering", () => {
  test("renders search results header with search term", async () => {
    render(<MainSearchResultPage {...defaultProps} />);

    expect(await screen.findByText("Results for:")).toBeInTheDocument();
    expect(await screen.findByText("test-search")).toBeInTheDocument();
  });

  test("renders search bar", async () => {
    render(<MainSearchResultPage {...defaultProps} />);

    expect(await screen.findByPlaceholderText(/search/i)).toBeInTheDocument();
  });
});
