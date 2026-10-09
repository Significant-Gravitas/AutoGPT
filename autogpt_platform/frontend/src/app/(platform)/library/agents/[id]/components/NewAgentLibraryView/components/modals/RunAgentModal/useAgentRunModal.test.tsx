import { getPostV1ExecuteGraphAgentMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { act, renderHook, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import type { ReactNode } from "react";
import { afterEach, describe, expect, test, vi } from "vitest";
import { useAgentRunModal } from "./useAgentRunModal";

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
  input_schema: { properties: {}, required: [] },
  credentials_input_schema: { properties: {}, required: [] },
  trigger_setup_info: null,
} as unknown as LibraryAgent;

const execution = {
  id: "run-1",
  graph_id: "graph-xyz",
} as GraphExecutionMeta;

afterEach(() => {
  toastMock.mockClear();
  sendDatafastEvent.mockClear();
  server.resetHandlers();
});

describe("useAgentRunModal", () => {
  test("a real run records the run_agent goal from the library surface", async () => {
    server.use(getPostV1ExecuteGraphAgentMockHandler(execution));
    const onRun = vi.fn();

    const { result } = renderHook(() => useAgentRunModal(agent, { onRun }), {
      wrapper,
    });

    act(() => {
      result.current.handleRun();
    });

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({ id: "run-1" }),
      );
    });
    expect(sendDatafastEvent).toHaveBeenCalledExactlyOnceWith("run_agent", {
      id: "graph-xyz",
      name: "My agent",
      surface: "library",
    });
  });

  test("a simulation goes through the same mutation but is not an activation", async () => {
    server.use(getPostV1ExecuteGraphAgentMockHandler(execution));
    const onRun = vi.fn();

    const { result } = renderHook(() => useAgentRunModal(agent, { onRun }), {
      wrapper,
    });

    act(() => {
      result.current.handleSimulate();
    });

    await waitFor(() => {
      expect(onRun).toHaveBeenCalledWith(
        expect.objectContaining({ id: "run-1" }),
      );
    });
    expect(sendDatafastEvent).not.toHaveBeenCalled();
  });
});
