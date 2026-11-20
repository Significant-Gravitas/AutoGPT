import {
  getGetV1GetExecutionDetailsMockHandler,
  getGetV1GetExecutionDetailsResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { server } from "@/mocks/mock-server";
import { renderHook, waitFor } from "@/tests/integrations/test-utils";
import {
  focusManager,
  QueryClient,
  QueryClientProvider,
} from "@tanstack/react-query";
import { ReactNode } from "react";
import { afterEach, expect, test, vi } from "vitest";
import { useSelectedRunView } from "../useSelectedRunView";

const POLL_MS = 1500;
const TEST_TIMEOUT_MS = 20_000;

afterEach(() => {
  focusManager.setFocused(undefined);
});

function wait(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

test(
  "polls a running run while the tab is visible and pauses while it is hidden",
  async () => {
    const detailRequests = vi.fn(() => ({
      ...getGetV1GetExecutionDetailsResponseMock(),
      id: "run-1",
      graph_id: "graph-1",
      status: AgentExecutionStatus.RUNNING,
      preset_id: null,
      node_executions: [],
    }));
    server.use(getGetV1GetExecutionDetailsMockHandler(detailRequests));
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    function wrapper({ children }: { children: ReactNode }) {
      return (
        <QueryClientProvider client={queryClient}>
          {children}
        </QueryClientProvider>
      );
    }

    renderHook(() => useSelectedRunView("graph-1", "run-1"), { wrapper });
    await waitFor(() => expect(detailRequests).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(detailRequests).toHaveBeenCalledTimes(2), {
      timeout: POLL_MS * 2,
    });

    focusManager.setFocused(false);
    const whileHidden = detailRequests.mock.calls.length;
    await wait(POLL_MS * 2);
    expect(detailRequests).toHaveBeenCalledTimes(whileHidden);

    focusManager.setFocused(true);
    await waitFor(
      () =>
        expect(detailRequests.mock.calls.length).toBeGreaterThan(whileHidden),
      { timeout: POLL_MS * 2 },
    );
  },
  TEST_TIMEOUT_MS,
);
