import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import type { GraphExecution } from "@/lib/autogpt-server-api/types";

type MessageHandler = (execution: GraphExecution) => void;

const messageHandlers = new Set<MessageHandler>();
const subscribeToGraphExecutions = vi.fn(() => Promise.resolve());
const onWebSocketConnect = vi.fn((handler: () => void) => {
  handler();
  return () => undefined;
});
const fakeAPI = {
  onWebSocketConnect,
  subscribeToGraphExecutions,
  connectWebSocket: vi.fn(() => Promise.resolve()),
  onWebSocketMessage: (_method: string, handler: MessageHandler) => {
    messageHandlers.add(handler);
    return () => messageHandlers.delete(handler);
  },
};

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => fakeAPI,
}));

import { useExecutionEvents } from "../useExecutionEvents";

function makeExecution(graphID: string, id: string) {
  return { id, graph_id: graphID } as unknown as GraphExecution;
}

function emit(execution: GraphExecution) {
  act(() => {
    messageHandlers.forEach((handler) => handler(execution));
  });
}

beforeEach(() => {
  messageHandlers.clear();
  subscribeToGraphExecutions.mockClear();
  onWebSocketConnect.mockClear();
});

describe("useExecutionEvents subscriptions", () => {
  test("a re-render with an equal but new graphIds array does not re-subscribe", () => {
    const { rerender } = renderHook(
      ({ graphIds }) => useExecutionEvents({ graphIds }),
      { initialProps: { graphIds: ["graph-a", "graph-b"] } },
    );
    expect(subscribeToGraphExecutions).toHaveBeenCalledTimes(2);

    rerender({ graphIds: ["graph-a", "graph-b"] });
    rerender({ graphIds: ["graph-a", "graph-b"] });

    expect(onWebSocketConnect).toHaveBeenCalledTimes(1);
    expect(subscribeToGraphExecutions).toHaveBeenCalledTimes(2);
  });

  test("a changed graph ID list subscribes again", () => {
    const { rerender } = renderHook(
      ({ graphIds }) => useExecutionEvents({ graphIds }),
      { initialProps: { graphIds: ["graph-a"] } },
    );
    rerender({ graphIds: ["graph-a", "graph-b"] });

    expect(subscribeToGraphExecutions.mock.calls).toEqual([
      ["graph-a"],
      ["graph-a"],
      ["graph-b"],
    ]);
  });

  test("only events for the subscribed graphs reach the handler", () => {
    const onExecutionUpdate = vi.fn();
    renderHook(() =>
      useExecutionEvents({ graphIds: ["graph-a"], onExecutionUpdate }),
    );

    emit(makeExecution("graph-b", "run-1"));
    emit(makeExecution("graph-a", "run-2"));

    expect(onExecutionUpdate.mock.calls).toEqual([
      [makeExecution("graph-a", "run-2")],
    ]);
  });
});

describe("useExecutionEvents throttleMs", () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });
  afterEach(() => {
    vi.useRealTimers();
  });

  test("without throttleMs every event is delivered", () => {
    const onExecutionUpdate = vi.fn();
    renderHook(() =>
      useExecutionEvents({ graphId: "graph-a", onExecutionUpdate }),
    );

    for (let index = 0; index < 5; index += 1) {
      emit(makeExecution("graph-a", `run-${index}`));
    }

    expect(onExecutionUpdate).toHaveBeenCalledTimes(5);
  });

  test("delivers the first event at once and the latest one when the window ends", () => {
    const onExecutionUpdate = vi.fn();
    renderHook(() =>
      useExecutionEvents({
        graphId: "graph-a",
        throttleMs: 2000,
        onExecutionUpdate,
      }),
    );

    for (let index = 0; index < 10; index += 1) {
      emit(makeExecution("graph-a", `run-${index}`));
    }
    expect(onExecutionUpdate.mock.calls).toEqual([
      [makeExecution("graph-a", "run-0")],
    ]);

    act(() => {
      vi.advanceTimersByTime(1999);
    });
    expect(onExecutionUpdate).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(1);
    });
    expect(onExecutionUpdate.mock.calls).toEqual([
      [makeExecution("graph-a", "run-0")],
      [makeExecution("graph-a", "run-9")],
    ]);

    act(() => {
      vi.advanceTimersByTime(10_000);
    });
    expect(onExecutionUpdate).toHaveBeenCalledTimes(2);
  });

  test("an event after a quiet window is delivered at once", () => {
    const onExecutionUpdate = vi.fn();
    renderHook(() =>
      useExecutionEvents({
        graphId: "graph-a",
        throttleMs: 2000,
        onExecutionUpdate,
      }),
    );

    emit(makeExecution("graph-a", "run-1"));
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    emit(makeExecution("graph-a", "run-2"));

    expect(onExecutionUpdate).toHaveBeenCalledTimes(2);
  });

  test("a pending event is delivered when the subscription is torn down", () => {
    const onExecutionUpdate = vi.fn();
    const { unmount } = renderHook(() =>
      useExecutionEvents({
        graphId: "graph-a",
        throttleMs: 2000,
        onExecutionUpdate,
      }),
    );

    emit(makeExecution("graph-a", "run-1"));
    emit(makeExecution("graph-a", "run-2"));
    unmount();

    expect(onExecutionUpdate.mock.calls).toEqual([
      [makeExecution("graph-a", "run-1")],
      [makeExecution("graph-a", "run-2")],
    ]);
    act(() => {
      vi.advanceTimersByTime(2000);
    });
    expect(onExecutionUpdate).toHaveBeenCalledTimes(2);
  });
});
