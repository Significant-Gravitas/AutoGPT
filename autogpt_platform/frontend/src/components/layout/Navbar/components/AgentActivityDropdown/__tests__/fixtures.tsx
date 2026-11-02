import { getGetV1ListAllExecutionsMockHandler200 } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { getGetV2ListLibraryAgentsInfiniteQueryKey } from "@/app/api/__generated__/endpoints/library/library";
import {
  getGetV2GetLibraryAgentResponseMock200,
  getGetV2ListLibraryAgentsResponseMock200,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import BackendAPI from "@/lib/autogpt-server-api/client";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { vi } from "vitest";
import { AccountMenuActivityRow } from "../../AccountMenu/components/AccountMenuActivityRow";

export const libraryKey = getGetV2ListLibraryAgentsInfiniteQueryKey({
  page: 1,
  page_size: 100,
  is_hidden: false,
});
export const libraryURL = "http://localhost:3000/api/proxy/api/library/agents";
export const lateGraphID = "abcdef01-late-workflow";
export const earlyAgent = makeAgent("early-graph", "Early workflow");
export const lateAgent = makeAgent(lateGraphID, "Late-page workflow");
export const firstPageAgents = [
  earlyAgent,
  ...Array.from({ length: 99 }, (_, i) =>
    makeAgent(`graph-${i}`, `Workflow ${i}`),
  ),
];

export function makeAgent(graphID: string, name: string) {
  return getGetV2GetLibraryAgentResponseMock200({
    id: `library-${graphID}`,
    graph_id: graphID,
    name,
    description: "Workflow description",
  });
}

export function execution(
  overrides: Partial<GraphExecutionMeta> = {},
): GraphExecutionMeta {
  return {
    id: "execution-1",
    user_id: "user-1",
    graph_id: lateGraphID,
    graph_version: 1,
    inputs: {},
    credential_inputs: {},
    nodes_input_masks: {},
    preset_id: null,
    status: "COMPLETED",
    started_at: new Date(Date.now() - 60000),
    ended_at: new Date(),
    stats: null,
    ...overrides,
  };
}

export function page(
  agents: LibraryAgent[],
  currentPage = 1,
  totalItems = 101,
) {
  return getGetV2ListLibraryAgentsResponseMock200({
    agents,
    pagination: {
      current_page: currentPage,
      page_size: 100,
      total_items: totalItems,
      total_pages: Math.ceil(totalItems / 100),
    },
  });
}

export function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

export function renderActivity(executions = [execution()]) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay: 0, gcTime: 0 } },
  });
  // The provider and event subscription remain real; only the socket transport is idle.
  vi.spyOn(BackendAPI.prototype, "connectWebSocket").mockResolvedValue();
  server.use(getGetV1ListAllExecutionsMockHandler200(executions));
  const view = render(
    <QueryClientProvider client={queryClient}>
      <AccountMenuActivityRow />
    </QueryClientProvider>,
  );
  fireEvent.click(screen.getByTestId("account-menu-activity-trigger"));
  return { ...view, queryClient };
}

export async function settleRequests() {
  await new Promise((resolve) => setTimeout(resolve, 120));
}
