import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";

import { useBackendAPI } from "@/lib/autogpt-server-api/context";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";
import { useExecutionEvents } from "../useExecutionEvents";

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: vi.fn(),
}));

vi.mock("@sentry/nextjs", () => ({ captureException: vi.fn() }));

function createFakeAPI() {
  const connectHandlers = new Set<() => void>();
  const messageHandlers = new Set<(execution: GraphExecution) => void>();
  return {
    subscribeToGraphExecutions: vi.fn(async () => undefined),
    connectWebSocket: vi.fn(async () => undefined),
    onWebSocketConnect: vi.fn((handler: () => void) => {
      connectHandlers.add(handler);
      return () => connectHandlers.delete(handler);
    }),
    onWebSocketMessage: vi.fn(
      (_method: string, handler: (execution: GraphExecution) => void) => {
        messageHandlers.add(handler);
        return () => messageHandlers.delete(handler);
      },
    ),
    /** What the client does on every socket open, the first and each reconnect. */
    open() {
      connectHandlers.forEach((handler) => handler());
    },
    emit(execution: { id: string; graph_id: string }) {
      messageHandlers.forEach((handler) =>
        handler(execution as unknown as GraphExecution),
      );
    },
  };
}

describe("useExecutionEvents", () => {
  let api: ReturnType<typeof createFakeAPI>;

  beforeEach(() => {
    api = createFakeAPI();
    vi.mocked(useBackendAPI).mockReturnValue(
      api as unknown as ReturnType<typeof useBackendAPI>,
    );
  });

  test("refetches once the first subscriptions are confirmed", async () => {
    // A run can finish between the page's initial fetch and the moment the
    // subscription is live; that completion is never delivered.
    const onSubscribed = vi.fn();
    let confirm: () => void = () => {};
    api.subscribeToGraphExecutions.mockImplementationOnce(
      () =>
        new Promise<undefined>(
          (resolve) => (confirm = () => resolve(undefined)),
        ),
    );
    renderHook(() => useExecutionEvents({ graphId: "graph-a", onSubscribed }));

    api.open();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(onSubscribed).not.toHaveBeenCalled();

    confirm();
    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(1));
  });

  test("resubscribes every graph after a reconnect, then asks for a refetch", async () => {
    const onSubscribed = vi.fn();
    renderHook(() =>
      useExecutionEvents({ graphIds: ["graph-a", "graph-b"], onSubscribed }),
    );

    api.open();
    expect(api.subscribeToGraphExecutions).toHaveBeenCalledTimes(2);
    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(1));

    // The server dropped every subscription with the old socket.
    api.open();
    expect(api.subscribeToGraphExecutions).toHaveBeenCalledTimes(4);
    expect(api.subscribeToGraphExecutions).toHaveBeenLastCalledWith("graph-b");
    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(2));
  });

  test("refetches only once the server confirmed the resubscriptions", async () => {
    const onSubscribed = vi.fn();
    renderHook(() => useExecutionEvents({ graphId: "graph-a", onSubscribed }));
    api.open();
    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(1));
    let confirm: () => void = () => {};
    api.subscribeToGraphExecutions.mockImplementationOnce(
      () =>
        new Promise<undefined>(
          (resolve) => (confirm = () => resolve(undefined)),
        ),
    );

    api.open();
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(onSubscribed).toHaveBeenCalledTimes(1);

    confirm();
    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(2));
  });

  test("still asks for a refetch when a resubscribe fails", async () => {
    const onSubscribed = vi.fn();
    renderHook(() => useExecutionEvents({ graphId: "graph-a", onSubscribed }));
    api.open();
    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(1));
    api.subscribeToGraphExecutions.mockRejectedValueOnce(new Error("closed"));
    vi.spyOn(console, "error").mockImplementation(() => undefined);

    api.open();

    await waitFor(() => expect(onSubscribed).toHaveBeenCalledTimes(2));
  });

  test("only forwards events for the graphs it watches", () => {
    const onExecutionUpdate = vi.fn();
    renderHook(() =>
      useExecutionEvents({ graphId: "graph-a", onExecutionUpdate }),
    );

    api.emit({ id: "run-1", graph_id: "graph-b" });
    api.emit({ id: "run-2", graph_id: "graph-a" });

    expect(onExecutionUpdate).toHaveBeenCalledTimes(1);
    expect(onExecutionUpdate.mock.calls[0][0].id).toBe("run-2");
  });
});
