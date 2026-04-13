import {
  render,
  screen,
  cleanup,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { PlatformCostContent } from "../components/PlatformCostContent";
import type { PlatformCostDashboard } from "@/app/api/__generated__/models/platformCostDashboard";
import type { PlatformCostLogsResponse } from "@/app/api/__generated__/models/platformCostLogsResponse";

// Mock the generated Orval hooks so tests don't hit the network
const mockUseGetDashboard = vi.fn();
const mockUseGetLogs = vi.fn();

vi.mock("@/app/api/__generated__/endpoints/admin/admin", () => ({
  useGetV2GetPlatformCostDashboard: (...args: unknown[]) =>
    mockUseGetDashboard(...args),
  useGetV2GetPlatformCostLogs: (...args: unknown[]) => mockUseGetLogs(...args),
}));

afterEach(() => {
  cleanup();
  mockUseGetDashboard.mockReset();
  mockUseGetLogs.mockReset();
});

const emptyDashboard: PlatformCostDashboard = {
  total_cost_microdollars: 0,
  total_requests: 0,
  total_users: 0,
  by_provider: [],
  by_user: [],
};

const emptyLogs: PlatformCostLogsResponse = {
  logs: [],
  pagination: {
    current_page: 1,
    page_size: 50,
    total_items: 0,
    total_pages: 0,
  },
};

const dashboardWithData: PlatformCostDashboard = {
  total_cost_microdollars: 5_000_000,
  total_requests: 100,
  total_users: 5,
  by_provider: [
    {
      provider: "openai",
      tracking_type: "tokens",
      total_cost_microdollars: 3_000_000,
      total_input_tokens: 50000,
      total_output_tokens: 20000,
      total_duration_seconds: 0,
      request_count: 60,
    },
    {
      provider: "google_maps",
      tracking_type: "per_run",
      total_cost_microdollars: 0,
      total_input_tokens: 0,
      total_output_tokens: 0,
      total_duration_seconds: 0,
      request_count: 40,
    },
  ],
  by_user: [
    {
      user_id: "user-1",
      email: "alice@example.com",
      total_cost_microdollars: 3_000_000,
      total_input_tokens: 50000,
      total_output_tokens: 20000,
      request_count: 60,
    },
  ],
};

const logsWithData: PlatformCostLogsResponse = {
  logs: [
    {
      id: "log-1",
      created_at: "2026-03-01T00:00:00Z" as unknown as Date,
      user_id: "user-1",
      email: "alice@example.com",
      graph_exec_id: "gx-123",
      node_exec_id: "nx-456",
      block_name: "LLMBlock",
      provider: "openai",
      tracking_type: "tokens",
      cost_microdollars: 5000,
      input_tokens: 100,
      output_tokens: 50,
      duration: 1.5,
      model: "gpt-4",
    },
  ],
  pagination: {
    current_page: 1,
    page_size: 50,
    total_items: 1,
    total_pages: 1,
  },
};

function renderComponent(searchParams = {}) {
  return render(<PlatformCostContent searchParams={searchParams} />);
}

describe("PlatformCostContent", () => {
  it("shows loading state initially", () => {
    mockUseGetDashboard.mockReturnValue({ data: undefined, isLoading: true });
    mockUseGetLogs.mockReturnValue({ data: undefined, isLoading: true });
    renderComponent();
    // Loading state renders Skeleton placeholders (animate-pulse divs) instead of content
    expect(screen.queryByText("Loading...")).toBeNull();
    // Summary cards and table content are not yet shown
    expect(screen.queryByText("Known Cost")).toBeNull();
  });

  it("renders empty dashboard", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: emptyLogs,
      isLoading: false,
    });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    // Verify the two summary cards that show $0.0000 — Known Cost and Estimated Total
    const zeroCostItems = screen.getAllByText("$0.0000");
    expect(zeroCostItems.length).toBe(2);
    expect(screen.getByText("No cost data yet")).toBeDefined();
  });

  it("renders dashboard with provider data", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("$5.0000")).toBeDefined();
    expect(screen.getByText("100")).toBeDefined();
    expect(screen.getByText("5")).toBeDefined();
    expect(screen.getByText("openai")).toBeDefined();
    expect(screen.getByText("google_maps")).toBeDefined();
  });

  it("renders tracking type badges", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("tokens")).toBeDefined();
    expect(screen.getByText("per_run")).toBeDefined();
  });

  it("shows error state on fetch failure", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Network error"),
    });
    mockUseGetLogs.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error("Network error"),
    });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("Network error")).toBeDefined();
  });

  it("renders tab buttons", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("By Provider")).toBeDefined();
    expect(screen.getByText("By User")).toBeDefined();
    expect(screen.getByText("Raw Logs")).toBeDefined();
  });

  it("renders summary cards with correct labels", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getAllByText("Known Cost").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Estimated Total")).toBeDefined();
    expect(screen.getByText("Total Requests")).toBeDefined();
    expect(screen.getByText("Active Users")).toBeDefined();
  });

  it("renders filter inputs", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("Start Date")).toBeDefined();
    expect(screen.getByText("End Date")).toBeDefined();
    expect(screen.getAllByText(/Provider/i).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("User ID")).toBeDefined();
    expect(screen.getByText("Apply")).toBeDefined();
  });

  it("renders by-user tab when specified", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent({ tab: "by-user" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("alice@example.com")).toBeDefined();
  });

  it("renders logs tab when specified", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent({ tab: "logs" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("LLMBlock")).toBeDefined();
    expect(screen.getByText("gpt-4")).toBeDefined();
  });

  it("renders no logs message when empty", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent({ tab: "logs" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("No logs found")).toBeDefined();
  });

  it("shows pagination when multiple pages", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    const multiPageLogs: PlatformCostLogsResponse = {
      logs: logsWithData.logs,
      pagination: {
        current_page: 1,
        page_size: 50,
        total_items: 200,
        total_pages: 4,
      },
    };
    mockUseGetLogs.mockReturnValue({
      data: multiPageLogs,
      isLoading: false,
    });
    renderComponent({ tab: "logs" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("Previous")).toBeDefined();
    expect(screen.getByText("Next")).toBeDefined();
    expect(screen.getByText(/Page 1 of 4/)).toBeDefined();
  });

  it("renders user table with unknown email", async () => {
    const dashWithNullEmail: PlatformCostDashboard = {
      ...dashboardWithData,
      by_user: [
        {
          user_id: "user-2",
          email: null,
          total_cost_microdollars: 1000,
          total_input_tokens: 100,
          total_output_tokens: 50,
          request_count: 5,
        },
      ],
    };
    mockUseGetDashboard.mockReturnValue({
      data: dashWithNullEmail,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent({ tab: "by-user" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("Unknown")).toBeDefined();
  });

  it("by-user tab content visible when tab=by-user param set", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent({ tab: "by-user" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("alice@example.com")).toBeDefined();
    // overview tab content should not be visible
    expect(screen.queryByText("openai")).toBeNull();
  });

  it("logs tab content visible when tab=logs param set", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: dashboardWithData,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logsWithData,
      isLoading: false,
    });
    renderComponent({ tab: "logs" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("LLMBlock")).toBeDefined();
    expect(screen.getByText("gpt-4")).toBeDefined();
  });

  it("renders log with null user as dash", async () => {
    const logWithNullUser: PlatformCostLogsResponse = {
      logs: [
        {
          id: "log-2",
          created_at: "2026-03-01T00:00:00Z" as unknown as Date,
          user_id: null,
          email: null,
          graph_exec_id: null,
          node_exec_id: null,
          block_name: "copilot:SDK",
          provider: "anthropic",
          tracking_type: "cost_usd",
          cost_microdollars: 15000,
          input_tokens: null,
          output_tokens: null,
          duration: null,
          model: "claude-opus-4-20250514",
        },
      ],
      pagination: {
        current_page: 1,
        page_size: 50,
        total_items: 1,
        total_pages: 1,
      },
    };
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({
      data: logWithNullUser,
      isLoading: false,
    });
    renderComponent({ tab: "logs" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("copilot:SDK")).toBeDefined();
    expect(screen.getByText("anthropic")).toBeDefined();
    // null email + null user_id renders as "-" in the User column; multiple
    // other cells (tokens, duration, session) also render "-", so use
    // getAllByText to avoid the single-match constraint.
    expect(screen.getAllByText("-").length).toBeGreaterThan(0);
  });
});
