import { expect, test } from "vitest";
import { render, screen } from "@/tests/test-utils";
import { MainMarkeplacePage } from "./MainMarketplacePage";

test("MainMarketplacePage", () => {
  render(<MainMarkeplacePage />);
  expect(screen.getByText("Featured Agents")).toBeDefined();
});