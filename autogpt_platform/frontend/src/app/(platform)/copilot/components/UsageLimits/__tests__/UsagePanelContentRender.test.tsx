import { render, screen } from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { UsagePanelContent, formatBytes } from "../UsagePanelContent";
import type { CoPilotUsagePublic } from "@/app/api/__generated__/models/coPilotUsagePublic";

const mockStorageData = vi.fn();
vi.mock("../useWorkspaceStorage", () => ({
  useWorkspaceStorage: () => mockStorageData(),
}));

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
  mockStorageData.mockReset();
  mockUseGetFlag.mockReset();
});

beforeEach(() => {
  mockStorageData.mockReturnValue({ data: undefined });
  mockUseGetFlag.mockReturnValue(true);
});

function makeUsage(
  overrides: Partial<{
    dailyPercent: number | null;
    weeklyPercent: number | null;
    tier: string;
  }> = {},
): CoPilotUsagePublic {
  const { dailyPercent = 5, weeklyPercent = 4, tier = "BASIC" } = overrides;
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
  } as CoPilotUsagePublic;
}

describe("formatBytes", () => {
  it.each([
    [0, "0 B"],
    [512, "512 B"],
    [1024, "1 KB"],
    [250 * 1024, "250 KB"],
    [1023 * 1024, "1023 KB"],
    [1000 * 1024, "1000 KB"],
    [1024 * 1024, "1 MB"],
    [250 * 1024 * 1024, "250 MB"],
    [1000 * 1024 * 1024, "1000 MB"],
    [1024 * 1024 * 1024, "1.0 GB"],
    [5 * 1024 * 1024 * 1024, "5.0 GB"],
    [15 * 1024 * 1024 * 1024, "15.0 GB"],
  ])("formats %d bytes as %s", (input, expected) => {
    expect(formatBytes(input)).toBe(expected);
  });
});

describe("UsagePanelContent", () => {
  it("renders 'No usage limits configured' when both windows are null", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: null, weeklyPercent: null })}
      />,
    );
    expect(screen.getByText("No usage limits configured")).toBeDefined();
  });

  it("still renders file storage when usage windows are null", () => {
    mockStorageData.mockReturnValue({
      data: {
        used_bytes: 100 * 1024 * 1024,
        limit_bytes: 250 * 1024 * 1024,
        used_percent: 40,
        file_count: 5,
      },
    });

    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: null, weeklyPercent: null })}
      />,
    );

    expect(screen.getByText("No usage limits configured")).toBeDefined();
    expect(screen.getByText("File storage")).toBeDefined();
  });

  it("never renders the legacy 'Reset daily limit' button", () => {
    render(<UsagePanelContent usage={makeUsage({ dailyPercent: 100 })} />);
    expect(screen.queryByText(/Reset daily limit/)).toBeNull();
  });

  it("renders 'Go to billing' when daily is exhausted and billing is enabled", () => {
    mockUseGetFlag.mockReturnValue(true);
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, weeklyPercent: 40 })}
      />,
    );
    expect(screen.getByText("Go to billing")).toBeDefined();
  });

  it("hides the 'Learn more about usage limits' link when 'Go to billing' button is shown", () => {
    mockUseGetFlag.mockReturnValue(true);
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, weeklyPercent: 40 })}
      />,
    );
    expect(screen.getByText("Go to billing")).toBeDefined();
    expect(screen.queryByText("Learn more about usage limits")).toBeNull();
  });

  it("does not render 'Go to billing' when showBillingLink is false", () => {
    mockUseGetFlag.mockReturnValue(true);
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, weeklyPercent: 40 })}
        showBillingLink={false}
      />,
    );
    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("does not render 'Go to billing' when billing flag is disabled", () => {
    mockUseGetFlag.mockReturnValue(false);
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, weeklyPercent: 40 })}
      />,
    );
    expect(screen.queryByText("Go to billing")).toBeNull();
  });

  it("still renders 'Go to billing' when both daily and weekly are exhausted", () => {
    render(
      <UsagePanelContent
        usage={makeUsage({ dailyPercent: 100, weeklyPercent: 100 })}
      />,
    );
    expect(screen.getByText("Go to billing")).toBeDefined();
  });

  it("renders percent used in the usage bar", () => {
    render(<UsagePanelContent usage={makeUsage({ dailyPercent: 25 })} />);
    expect(screen.getByText("25% used")).toBeDefined();
  });

  it("renders '<1% used' when usage is greater than 0 but rounds to 0", () => {
    render(<UsagePanelContent usage={makeUsage({ dailyPercent: 0.3 })} />);
    expect(screen.getByText("<1% used")).toBeDefined();
  });

  it("renders file storage bar when workspace data is available", () => {
    mockStorageData.mockReturnValue({
      data: {
        used_bytes: 100 * 1024 * 1024,
        limit_bytes: 250 * 1024 * 1024,
        used_percent: 40,
        file_count: 5,
      },
    });

    render(<UsagePanelContent usage={makeUsage()} />);
    expect(screen.getByText("File storage")).toBeDefined();
    expect(screen.getByText(/100 MB of 250 MB/)).toBeDefined();
    expect(screen.getByText(/5 files/)).toBeDefined();
  });

  it("hides file storage bar when no workspace data", () => {
    mockStorageData.mockReturnValue({ data: undefined });

    render(<UsagePanelContent usage={makeUsage()} />);
    expect(screen.queryByText("File storage")).toBeNull();
  });

  it("hides file storage bar when limit is zero", () => {
    mockStorageData.mockReturnValue({
      data: {
        used_bytes: 0,
        limit_bytes: 0,
        used_percent: 0,
        file_count: 0,
      },
    });

    render(<UsagePanelContent usage={makeUsage()} />);
    expect(screen.queryByText("File storage")).toBeNull();
  });

  it("shows orange bar when storage usage is at or above 80%", () => {
    mockStorageData.mockReturnValue({
      data: {
        used_bytes: 210 * 1024 * 1024,
        limit_bytes: 250 * 1024 * 1024,
        used_percent: 84,
        file_count: 3,
      },
    });

    render(<UsagePanelContent usage={makeUsage()} />);
    expect(screen.getByText("File storage")).toBeDefined();
    expect(screen.getByText("84% used")).toBeDefined();
  });

  it("shows singular 'file' for single file", () => {
    mockStorageData.mockReturnValue({
      data: {
        used_bytes: 1024,
        limit_bytes: 250 * 1024 * 1024,
        used_percent: 0,
        file_count: 1,
      },
    });

    render(<UsagePanelContent usage={makeUsage()} />);
    expect(screen.getByText(/1 file$/)).toBeDefined();
  });

  it("shows storage '<1% used' when usage is tiny", () => {
    mockStorageData.mockReturnValue({
      data: {
        used_bytes: 100,
        limit_bytes: 250 * 1024 * 1024,
        used_percent: 0.001,
        file_count: 1,
      },
    });

    render(<UsagePanelContent usage={makeUsage()} />);
    expect(screen.getByText("File storage")).toBeDefined();
  });

  it("renders header with tier label", () => {
    render(<UsagePanelContent usage={makeUsage({ tier: "PRO" })} />);
    expect(screen.getByText("Pro plan")).toBeDefined();
  });

  it("hides header when showHeader is false", () => {
    render(<UsagePanelContent usage={makeUsage()} showHeader={false} />);
    expect(screen.queryByText("Usage limits")).toBeNull();
  });
});
