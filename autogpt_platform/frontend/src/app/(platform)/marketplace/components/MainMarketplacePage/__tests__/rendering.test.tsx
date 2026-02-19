import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";

describe("MainMarketplacePage - Rendering", () => {
  test("renders hero section with search bar", async () => {
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured agents", { exact: false }),
    ).toBeInTheDocument();

    expect(screen.getByPlaceholderText(/search/i)).toBeInTheDocument();
  });

  test("renders featured agents section", async () => {
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured agents", { exact: false }),
    ).toBeInTheDocument();
  });

  test("renders top agents section", async () => {
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Top Agents", { exact: false }),
    ).toBeInTheDocument();
  });

  test("renders featured creators section", async () => {
    render(<MainMarkeplacePage />);

    expect(
      await screen.findByText("Featured creators", { exact: false }),
    ).toBeInTheDocument();
  });
});
