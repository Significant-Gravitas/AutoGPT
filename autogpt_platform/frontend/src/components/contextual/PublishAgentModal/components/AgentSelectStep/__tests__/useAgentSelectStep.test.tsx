import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { ReactNode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { MyAgentsSortBy } from "@/app/api/__generated__/models/myAgentsSortBy";

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isLoggedIn: true }),
}));

const getMyAgentsMock = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/store/store", () => ({
  useGetV2GetMyAgents: (
    params: unknown,
    options: { query?: { select?: (res: unknown) => unknown } } = {},
  ) => {
    const res = getMyAgentsMock(params);
    const select = options.query?.select;
    const selected = select ? select(res) : res;
    return {
      data: selected,
      isLoading: false,
      isFetching: false,
      error: null,
    };
  },
}));

import { useAgentSelectStep } from "../useAgentSelectStep";

function buildAgent(idx: number) {
  return {
    graph_id: `graph-${idx}`,
    graph_version: 1,
    agent_name: `Agent ${idx}`,
    description: `Description ${idx}`,
    last_edited: new Date(`2026-05-${String(idx).padStart(2, "0")}T00:00:00Z`),
    agent_image: null,
    recommended_schedule_cron: null,
  };
}

function mockResponse(total: number, page: number, pageSize: number) {
  const start = (page - 1) * pageSize;
  const agents = Array.from(
    { length: Math.min(pageSize, total - start) },
    (_, i) => buildAgent(start + i + 1),
  );
  return {
    status: 200,
    data: {
      agents,
      pagination: {
        current_page: page,
        total_items: total,
        total_pages: Math.ceil(total / pageSize),
        page_size: pageSize,
      },
    },
  };
}

function wrap(children: ReactNode) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

describe("useAgentSelectStep", () => {
  const onSelect = vi.fn();
  const onNext = vi.fn();

  beforeEach(() => {
    getMyAgentsMock.mockReset();
    onSelect.mockReset();
    onNext.mockReset();
    getMyAgentsMock.mockImplementation((params: { page?: number }) =>
      mockResponse(47, params?.page ?? 1, 10),
    );
  });

  function setup() {
    return renderHook(() => useAgentSelectStep({ onSelect, onNext }), {
      wrapper: ({ children }) => wrap(children),
    });
  }

  it("exposes the agents from the API mapped to the view-model", async () => {
    const { result } = setup();
    await waitFor(() => expect(result.current.myAgents).toHaveLength(10));
    expect(result.current.myAgents[0]).toMatchObject({
      id: "graph-1",
      version: 1,
      name: "Agent 1",
    });
    expect(result.current.totalPages).toBe(5);
    expect(result.current.totalItems).toBe(47);
    expect(result.current.pageSize).toBe(10);
    expect(result.current.sortBy).toBe(MyAgentsSortBy.most_recent);
  });

  it("handleAgentClick selects an agent and forwards to onSelect", async () => {
    const { result } = setup();
    await waitFor(() =>
      expect(result.current.myAgents.length).toBeGreaterThan(0),
    );
    act(() => {
      result.current.handleAgentClick("Agent 1", "graph-1", 1);
    });
    expect(result.current.selectedAgentId).toBe("graph-1");
    expect(result.current.isNextDisabled).toBe(false);
    expect(onSelect).toHaveBeenCalledWith("graph-1", 1);
  });

  it("handleNext invokes onNext only when an agent is selected", async () => {
    const { result } = setup();
    await waitFor(() =>
      expect(result.current.myAgents.length).toBeGreaterThan(0),
    );

    act(() => {
      result.current.handleNext();
    });
    expect(onNext).not.toHaveBeenCalled();

    act(() => {
      result.current.handleAgentClick("Agent 1", "graph-1", 1);
    });
    act(() => {
      result.current.handleNext();
    });
    expect(onNext).toHaveBeenCalledWith(
      "graph-1",
      1,
      expect.objectContaining({
        name: "Agent 1",
        description: "Description 1",
      }),
    );
  });

  it("goToPage updates page and pageDirection, ignoring out-of-range values", async () => {
    const { result } = setup();
    await waitFor(() => expect(result.current.totalPages).toBe(5));

    act(() => {
      result.current.goToPage(2);
    });
    expect(result.current.page).toBe(2);
    expect(result.current.pageDirection).toBe(1);

    act(() => {
      result.current.goToPage(1);
    });
    expect(result.current.page).toBe(1);
    expect(result.current.pageDirection).toBe(-1);

    act(() => {
      result.current.goToPage(0); // below min, ignored
    });
    expect(result.current.page).toBe(1);

    act(() => {
      result.current.goToPage(99); // above max, ignored
    });
    expect(result.current.page).toBe(1);
  });

  it("handleSortChange resets to page 1", async () => {
    const { result } = setup();
    await waitFor(() => expect(result.current.totalPages).toBe(5));

    act(() => {
      result.current.goToPage(3);
    });
    expect(result.current.page).toBe(3);

    act(() => {
      result.current.handleSortChange(MyAgentsSortBy.name);
    });
    expect(result.current.sortBy).toBe(MyAgentsSortBy.name);
    await waitFor(() => expect(result.current.page).toBe(1));
  });
});
