import { http, HttpResponse, type JsonBodyType } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { UsagePopover } from "../UsagePopover";

const mockUseGetFlag = vi.fn();
vi.mock("@/services/feature-flags/use-get-flag", async () => {
  const actual = await vi.importActual<
    typeof import("@/services/feature-flags/use-get-flag")
  >("@/services/feature-flags/use-get-flag");
  return {
    ...actual,
    useGetFlag: (flag: unknown) => mockUseGetFlag(flag),
  };
});

afterEach(() => {
  mockUseGetFlag.mockReset();
});

beforeEach(() => {
  mockUseGetFlag.mockReturnValue(true);
});

vi.mock("../../StorageBar", () => ({
  StorageBar: () => null,
}));

vi.mock("@/components/molecules/Popover/Popover", () => {
  function Popover({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverTrigger({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  function PopoverContent({ children }: { children: React.ReactNode }) {
    return <div>{children}</div>;
  }
  return { Popover, PopoverTrigger, PopoverContent };
});

interface UsageOverrides {
  dailyPercent?: number | null;
  weeklyPercent?: number | null;
  tier?: string | null;
}

function makeUsage({
  dailyPercent = 5,
  weeklyPercent = 4,
  tier = "BASIC",
}: UsageOverrides = {}) {
  const future = new Date(Date.now() + 3600 * 1000).toISOString();
  return {
    daily:
      dailyPercent === null
        ? null
        : { percent_used: dailyPercent, resets_at: future },
    weekly:
      weeklyPercent === null
        ? null
        : { percent_used: weeklyPercent, resets_at: future },
    tier,
  };
}

function mockUsageResponse(body: JsonBodyType) {
  server.use(http.get("*/api/chat/usage", () => HttpResponse.json(body)));
}

describe("UsagePopover", () => {
  it("renders nothing on the first paint while data is loading", () => {
    mockUsageResponse(makeUsage({ dailyPercent: 50 }));
    const { container } = render(<UsagePopover />);
    expect(container.innerHTML).toBe("");
  });

  it("renders nothing when no limits are configured", async () => {
    mockUsageResponse(makeUsage({ dailyPercent: null, weeklyPercent: null }));
    const { container } = render(<UsagePopover />);
    await waitFor(() => expect(container.innerHTML).toBe(""));
  });

  it("renders the trigger button and panel when limits exist", async () => {
    mockUsageResponse(makeUsage({ dailyPercent: 50 }));
    render(<UsagePopover />);

    expect(
      await screen.findByRole("button", { name: /usage limits/i }),
    ).toBeDefined();
    expect(screen.getByText("Usage limits")).toBeDefined();
    expect(screen.getByText("Today")).toBeDefined();
    expect(screen.getByText("This week")).toBeDefined();
    expect(screen.getByText("50% used")).toBeDefined();
  });

  it("shows only the weekly bar when daily is null", async () => {
    mockUsageResponse(makeUsage({ dailyPercent: null, weeklyPercent: 50 }));
    render(<UsagePopover />);

    expect(await screen.findByText("This week")).toBeDefined();
    expect(screen.queryByText("Today")).toBeNull();
  });

  it("caps the bar width at 100% when over the limit", async () => {
    mockUsageResponse(makeUsage({ dailyPercent: 150 }));
    render(<UsagePopover />);

    const dailyBar = await screen.findByRole("progressbar", {
      name: /today usage/i,
    });
    expect(dailyBar.getAttribute("aria-valuenow")).toBe("100");
  });

  it("renders the tier label", async () => {
    mockUsageResponse(makeUsage({ tier: "PRO" }));
    render(<UsagePopover />);

    expect(await screen.findByText("Pro plan")).toBeDefined();
  });

  it("never renders the 'Go to billing' button (handled by the card)", async () => {
    mockUsageResponse(makeUsage({ dailyPercent: 100 }));
    render(<UsagePopover />);

    expect(await screen.findByText("Today")).toBeDefined();
    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("renders a 'Manage billing' link when billing is enabled", async () => {
    mockUsageResponse(makeUsage());
    render(<UsagePopover />);

    const link = (await screen.findByText("Manage billing")).closest("a");
    expect(link).not.toBeNull();
    expect(link?.getAttribute("href")).toBe("/settings/billing");
  });

  it("hides the 'Manage billing' link when billing is disabled at the platform level", async () => {
    mockUseGetFlag.mockReturnValue(false);
    mockUsageResponse(makeUsage());
    render(<UsagePopover />);

    expect(await screen.findByText("Today")).toBeDefined();
    expect(screen.queryByText("Manage billing")).toBeNull();
  });
});
