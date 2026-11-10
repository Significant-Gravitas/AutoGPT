import { render } from "@/tests/integrations/test-utils";
import { describe, expect, it, vi } from "vitest";
import { PlatformCostContent } from "../../platform-costs/components/PlatformCostContent";
import SpendingDashboardPage from "../../spending/page";
import { getSpendingSearchHref, getUserPlatformCostsHref } from "../helpers";

// The user page links out rather than summarising; these pin that each target
// page filters by the parameters the link carries.

const USER_ID = "5d0a1c2e-0000-4000-8000-000000000001";

const mockUseGetDashboard = vi.hoisted(() =>
  vi.fn(() => ({ data: undefined, isLoading: true, error: null })),
);
const mockGrantHistory = vi.hoisted(() => vi.fn(() => null));

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2GetPlatformCostDashboard: mockUseGetDashboard,
  useGetV2GetPlatformCostLogs: () => ({
    data: undefined,
    isLoading: true,
    error: null,
  }),
}));

vi.mock("@/lib/withRoleAccess", () => ({
  withRoleAccess: async () => async (Component: unknown) => Component,
}));

vi.mock("../../spending/components/AdminUserGrantHistory", () => ({
  AdminUserGrantHistory: mockGrantHistory,
}));

vi.mock("../../spending/components/ExportCreditTransactionsButton", () => ({
  ExportCreditTransactionsButton: () => null,
}));

vi.mock("../../spending/components/ExportCopilotUsageButton", () => ({
  ExportCopilotUsageButton: () => null,
}));

describe("the user page's money links", () => {
  it("open platform costs filtered to the user", () => {
    render(
      <PlatformCostContent
        searchParams={paramsOf(getUserPlatformCostsHref(USER_ID))}
      />,
    );

    expect(mockUseGetDashboard).toHaveBeenCalledWith(
      expect.objectContaining({ user_id: USER_ID }),
      expect.anything(),
    );
  });

  it("open spending history filtered to the user", async () => {
    const page = await SpendingDashboardPage({
      searchParams: Promise.resolve(paramsOf(getSpendingSearchHref(USER_ID))),
    });
    render(page);

    expect(mockGrantHistory).toHaveBeenCalledWith(
      expect.objectContaining({ initialSearch: USER_ID }),
      undefined,
    );
  });
});

function paramsOf(href: string) {
  return Object.fromEntries(new URL(href, "http://localhost").searchParams);
}
