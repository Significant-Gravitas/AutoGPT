import { describe, expect, it, vi } from "vitest";

const mockGetDashboard = vi.fn();
const mockGetLogs = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  getV2GetPlatformCostDashboard: (...args: unknown[]) =>
    mockGetDashboard(...args),
  getV2GetPlatformCostLogs: (...args: unknown[]) => mockGetLogs(...args),
}));

import { getPlatformCostDashboard, getPlatformCostLogs } from "../actions";

describe("getPlatformCostDashboard", () => {
  it("returns data on success", async () => {
    const mockData = { total_cost_microdollars: 1000, total_requests: 5 };
    mockGetDashboard.mockResolvedValue({ status: 200, data: mockData });
    const result = await getPlatformCostDashboard();
    expect(result).toEqual(mockData);
  });

  it("returns undefined on non-200", async () => {
    mockGetDashboard.mockResolvedValue({ status: 401 });
    const result = await getPlatformCostDashboard();
    expect(result).toBeUndefined();
  });

  it("passes filter params to API", async () => {
    mockGetDashboard.mockReset();
    mockGetDashboard.mockResolvedValue({ status: 200, data: {} });
    await getPlatformCostDashboard({
      start: "2026-01-01T00:00:00",
      end: "2026-06-01T00:00:00",
      provider: "openai",
      user_id: "user-1",
    });
    expect(mockGetDashboard).toHaveBeenCalledTimes(1);
    const params = mockGetDashboard.mock.calls[0][0];
    expect(params.start).toBe("2026-01-01T00:00:00");
    expect(params.end).toBe("2026-06-01T00:00:00");
    expect(params.provider).toBe("openai");
    expect(params.user_id).toBe("user-1");
  });

  it("passes undefined for empty filter strings", async () => {
    mockGetDashboard.mockReset();
    mockGetDashboard.mockResolvedValue({ status: 200, data: {} });
    await getPlatformCostDashboard({
      start: "",
      provider: "",
      user_id: "",
    });
    expect(mockGetDashboard).toHaveBeenCalledTimes(1);
    const params = mockGetDashboard.mock.calls[0][0];
    expect(params.start).toBeUndefined();
    expect(params.provider).toBeUndefined();
    expect(params.user_id).toBeUndefined();
  });
});

describe("getPlatformCostLogs", () => {
  it("returns data on success", async () => {
    const mockData = { logs: [], pagination: { current_page: 1 } };
    mockGetLogs.mockResolvedValue({ status: 200, data: mockData });
    const result = await getPlatformCostLogs();
    expect(result).toEqual(mockData);
  });

  it("passes page and page_size", async () => {
    mockGetLogs.mockReset();
    mockGetLogs.mockResolvedValue({ status: 200, data: { logs: [] } });
    await getPlatformCostLogs({ page: 3, page_size: 25 });
    expect(mockGetLogs).toHaveBeenCalledTimes(1);
    const params = mockGetLogs.mock.calls[0][0];
    expect(params.page).toBe(3);
    expect(params.page_size).toBe(25);
  });

  it("passes start date string through to API", async () => {
    mockGetLogs.mockReset();
    mockGetLogs.mockResolvedValue({ status: 200, data: { logs: [] } });
    await getPlatformCostLogs({ start: "2026-03-01T00:00:00" });
    expect(mockGetLogs).toHaveBeenCalledTimes(1);
    const params = mockGetLogs.mock.calls[0][0];
    expect(params.start).toBe("2026-03-01T00:00:00");
  });
});
