import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";

// Bypass server-side admin auth wrapper.
vi.mock("@/lib/withRoleAccess", () => ({
  withRoleAccess: () =>
    Promise.resolve((Component: React.ComponentType) =>
      Promise.resolve(Component),
    ),
}));

// The button owns the mutation hook; stub it so the page test stays focused on
// the dashboard layout.
vi.mock("../components/GenerateTestDataButton", () => ({
  GenerateTestDataButton: () => (
    <button type="button">Generate Test Data</button>
  ),
}));

import TestDataDashboardPage from "../page";

describe("TestDataDashboardPage", () => {
  it("renders the dashboard heading and script descriptions", async () => {
    render(await TestDataDashboardPage());

    expect(screen.getByText("Test Data Generation")).toBeDefined();
    expect(screen.getByText("Available Script Types:")).toBeDefined();
    expect(screen.getByText(/E2E Test Data:/i)).toBeDefined();
    expect(screen.getByText(/Full Test Data:/i)).toBeDefined();
    expect(screen.getByText("What data is created?")).toBeDefined();
  });

  it("renders the generate action", async () => {
    render(await TestDataDashboardPage());

    expect(
      screen.getByRole("button", { name: "Generate Test Data" }),
    ).toBeDefined();
  });
});
