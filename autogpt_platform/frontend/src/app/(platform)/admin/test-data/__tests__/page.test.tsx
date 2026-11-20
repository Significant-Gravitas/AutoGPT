import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as testDataHelpers from "../helpers";

// The shared next/navigation mock has no notFound(); re-declare it here with
// the same router surface plus the export this page depends on.
const notFoundMock = vi.hoisted(() => vi.fn());
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/admin/test-data",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

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
  afterEach(() => {
    vi.restoreAllMocks();
    notFoundMock.mockClear();
  });

  it("404s outside a local stack instead of rendering a dead-end page", async () => {
    vi.spyOn(testDataHelpers, "isTestDataSurfaceEnabled").mockReturnValue(
      false,
    );

    await expect(TestDataDashboardPage()).rejects.toThrow("NEXT_NOT_FOUND");
    expect(notFoundMock).toHaveBeenCalled();
  });

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
