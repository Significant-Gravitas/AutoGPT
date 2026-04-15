import {
  render,
  screen,
  cleanup,
  waitFor,
  fireEvent,
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
  total_input_tokens: 0,
  total_output_tokens: 0,
  avg_input_tokens_per_request: 0,
  avg_output_tokens_per_request: 0,
  avg_cost_microdollars_per_request: 0,
  cost_p50_microdollars: 0,
  cost_p75_microdollars: 0,
  cost_p95_microdollars: 0,
  cost_p99_microdollars: 0,
  cost_buckets: [],
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
  total_input_tokens: 150000,
  total_output_tokens: 60000,
  avg_input_tokens_per_request: 2500,
  avg_output_tokens_per_request: 1000,
  avg_cost_microdollars_per_request: 83333,
  cost_p50_microdollars: 50000,
  cost_p75_microdollars: 100000,
  cost_p95_microdollars: 250000,
  cost_p99_microdollars: 500000,
  cost_buckets: [
    { bucket: "$0-0.50", count: 80 },
    { bucket: "$0.50-1", count: 15 },
    { bucket: "$1-2", count: 5 },
  ],
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
      cost_bearing_request_count: 40,
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
    // Known Cost and Estimated Total cards render $0.0000
    // "Known Cost" appears in both the SummaryCard and the ProviderTable header
    expect(screen.getAllByText("Known Cost").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Estimated Total")).toBeDefined();
    // All cost summary cards (Known Cost, Estimated Total, Avg Cost,
    // Typical/Upper/High/Peak Cost) show $0.0000
    const zeroCostItems = screen.getAllByText("$0.0000");
    expect(zeroCostItems.length).toBe(7);
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
    // "5" appears in multiple places (Active Users card + bucket count),
    // so verify at least one element renders it.
    expect(screen.getAllByText("5").length).toBeGreaterThanOrEqual(1);
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
    // Original 4 cards
    expect(screen.getAllByText("Known Cost").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("Estimated Total")).toBeDefined();
    expect(screen.getByText("Total Requests")).toBeDefined();
    expect(screen.getByText("Active Users")).toBeDefined();
    // New average/token cards
    expect(screen.getByText("Avg Cost / Request")).toBeDefined();
    expect(screen.getByText("Avg Input Tokens")).toBeDefined();
    expect(screen.getByText("Avg Output Tokens")).toBeDefined();
    expect(screen.getByText("Total Tokens")).toBeDefined();
    // Percentile cards (friendlier labels)
    expect(screen.getByText("Typical Cost (P50)")).toBeDefined();
    expect(screen.getByText("Upper Cost (P75)")).toBeDefined();
    expect(screen.getByText("High Cost (P95)")).toBeDefined();
    expect(screen.getByText("Peak Cost (P99)")).toBeDefined();
  });

  it("renders cost distribution buckets", async () => {
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
    expect(screen.getByText("Cost Distribution by Bucket")).toBeDefined();
    expect(screen.getByText("$0-0.50")).toBeDefined();
    expect(screen.getByText("$0.50-1")).toBeDefined();
    expect(screen.getByText("$1-2")).toBeDefined();
    expect(screen.getByText("80")).toBeDefined();
    expect(screen.getByText("15")).toBeDefined();
  });

  it("renders new summary card values from fixture data", async () => {
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
    // Avg Input Tokens: 2500 formatted
    expect(screen.getByText("2,500")).toBeDefined();
    // Avg Output Tokens: 1000 formatted
    expect(screen.getByText("1,000")).toBeDefined();
    // P50 cost: 50000 microdollars = $0.0500
    expect(screen.getByText("$0.0500")).toBeDefined();
  });

  it("renders user table avg cost column with fixture data", async () => {
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
    // User table should show Avg Cost / Req header
    expect(screen.getByText("Avg Cost / Req")).toBeDefined();
    // Input/Output token columns
    expect(screen.getByText("Input Tokens")).toBeDefined();
    expect(screen.getByText("Output Tokens")).toBeDefined();
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

  it("renders execution ID filter input", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    expect(screen.getByText("Execution ID")).toBeDefined();
    expect(screen.getByPlaceholderText("Filter by execution")).toBeDefined();
  });

  it("pre-fills execution ID filter from searchParams", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent({ graph_exec_id: "exec-123" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    const input = screen.getByPlaceholderText(
      "Filter by execution",
    ) as HTMLInputElement;
    expect(input.value).toBe("exec-123");
  });

  it("clears execution ID input on Clear click", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent({ graph_exec_id: "exec-123" });
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    fireEvent.click(screen.getByText("Clear"));
    const input = screen.getByPlaceholderText(
      "Filter by execution",
    ) as HTMLInputElement;
    expect(input.value).toBe("");
  });

  it("passes execution ID to filter on Apply click", async () => {
    mockUseGetDashboard.mockReturnValue({
      data: emptyDashboard,
      isLoading: false,
    });
    mockUseGetLogs.mockReturnValue({ data: emptyLogs, isLoading: false });
    renderComponent();
    await waitFor(() =>
      expect(document.querySelector(".animate-pulse")).toBeNull(),
    );
    const input = screen.getByPlaceholderText(
      "Filter by execution",
    ) as HTMLInputElement;
    fireEvent.change(input, { target: { value: "exec-abc" } });
    expect(input.value).toBe("exec-abc");
    fireEvent.click(screen.getByText("Apply"));
    // After apply, the input still holds the typed value
    expect(input.value).toBe("exec-abc");
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
