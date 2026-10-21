import { getPostV1ExecuteGraphAgentMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useSelectedRunActions } from "./useSelectedRunActions";

function wrapper({ children }: { children: ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const sendDatafastEvent = vi.hoisted(() => vi.fn());
vi.mock("@/services/analytics", () => ({
  analytics: { sendDatafastEvent },
}));

const toastMock = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    useToast: () => ({ toast: toastMock }),
  };
});

const agent = {
  id: "lib-1",
  graph_id: "graph-xyz",
  graph_version: 1,
  name: "My agent",
} as unknown as LibraryAgent;

const run = {
  id: "run-0",
  graph_id: "graph-xyz",
  graph_version: 1,
  status: "COMPLETED",
  inputs: { topic: "news" },
  credential_inputs: {},
} as unknown as GraphExecution;

afterEach(() => {
  toastMock.mockClear();
  sendDatafastEvent.mockClear();
  server.resetHandlers();
});

describe("useSelectedRunActions", () => {
  test("handleRunAgain records a run_agent goal from the rerun surface", async () => {
    server.use(
      getPostV1ExecuteGraphAgentMockHandler({
        id: "run-1",
        graph_id: "graph-xyz",
      } as GraphExecutionMeta),
    );
    const onSelectRun = vi.fn();

    const { result } = renderHook(
      () =>
        useSelectedRunActions({
          agentGraphId: "graph-xyz",
          run,
          agent,
          onSelectRun,
        }),
      { wrapper },
    );

    await act(async () => {
      await result.current.handleRunAgain();
    });

    await waitFor(() => {
      expect(onSelectRun).toHaveBeenCalledWith("run-1");
    });
    expect(sendDatafastEvent).toHaveBeenCalledExactlyOnceWith("run_agent", {
      id: "graph-xyz",
      name: "My agent",
      surface: "rerun",
    });
  });

  test("handleRunAgain without a run surfaces a destructive toast and no goal", async () => {
    const { result } = renderHook(
      () => useSelectedRunActions({ agentGraphId: "graph-xyz", agent }),
      { wrapper },
    );

    await act(async () => {
      await result.current.handleRunAgain();
    });

    expect(toastMock).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Run not found",
        variant: "destructive",
      }),
    );
    expect(sendDatafastEvent).not.toHaveBeenCalled();
  });
});
