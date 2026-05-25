import { getGetV1ListAllExecutionsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { AgentListSection } from "../components/AgentListSection";
import { ExecutionListSection } from "../components/ExecutionListSection";

vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
    useSearchParams: () => new URLSearchParams(),
    usePathname: () => "/library",
  };
});

afterEach(() => {
  server.resetHandlers();
});

function makeAgent(overrides: Partial<LibraryAgent> = {}): LibraryAgent {
  return {
    id: "lib-1",
    graph_id: "g-1",
    name: "Agent One",
    image_url: null,
    has_external_trigger: false,
    is_scheduled: false,
    next_scheduled_run: null,
    recommended_schedule_cron: null,
    ...overrides,
  } as unknown as LibraryAgent;
}

function makeRunningExecution(graphID: string, id: string): GraphExecutionMeta {
  return {
    id,
    graph_id: graphID,
    graph_version: 1,
    status: "RUNNING",
    started_at: new Date().toISOString(),
    ended_at: null,
    stats: { activity_status: "Crunching data" },
  } as unknown as GraphExecutionMeta;
}

describe("ExecutionListSection", () => {
  it("renders running items and toggles the full list with 'Show all'", async () => {
    const agents = Array.from({ length: 7 }, (_, i) =>
      makeAgent({ id: `lib-${i}`, graph_id: `g-${i}`, name: `Agent ${i}` }),
    );
    const executions = agents.map((a, i) =>
      makeRunningExecution(a.graph_id, `exec-${i}`),
    );
    server.use(getGetV1ListAllExecutionsMockHandler200(executions));

    render(<ExecutionListSection activeTab="running" agents={agents} />);

    // 7 running items, but only MAX_VISIBLE (6) shown until expanded.
    await waitFor(() => {
      expect(screen.getAllByText("Crunching data").length).toBe(6);
    });
    const showAll = screen.getByRole("button", { name: /show all \(7\)/i });

    fireEvent.click(showAll);
    expect(screen.getAllByText("Crunching data").length).toBe(7);
    expect(screen.getByRole("button", { name: /collapse/i })).toBeDefined();
  });
});

describe("AgentListSection", () => {
  it("lists listening agents with their status label and toggles 'Show all'", async () => {
    const agents = Array.from({ length: 7 }, (_, i) =>
      makeAgent({
        id: `lib-${i}`,
        graph_id: `g-${i}`,
        name: `Listener ${i}`,
        has_external_trigger: true,
      }),
    );
    // No executions → status falls through to the config-derived "listening".
    server.use(getGetV1ListAllExecutionsMockHandler200([]));

    render(<AgentListSection activeTab="listening" agents={agents} />);

    await waitFor(() => {
      expect(screen.getAllByText("Waiting for trigger event").length).toBe(6);
    });

    fireEvent.click(screen.getByRole("button", { name: /show all \(7\)/i }));
    expect(screen.getAllByText("Waiting for trigger event").length).toBe(7);
  });

  it("shows the empty message when no agent matches the tab", () => {
    server.use(getGetV1ListAllExecutionsMockHandler200([]));
    render(<AgentListSection activeTab="idle" agents={[]} />);
    expect(screen.getByText("No idle agents")).toBeDefined();
  });
});
