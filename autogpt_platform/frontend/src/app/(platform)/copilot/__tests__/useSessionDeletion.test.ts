import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, renderHook } from "@testing-library/react";
import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (...args: unknown[]) => mockToast(...args),
}));

const setSessionIdMock = vi.fn();
let currentSessionId: string | null = "s1";
vi.mock("nuqs", () => ({
  parseAsString: {},
  useQueryState: () => [
    currentSessionId,
    (next: string | null) => {
      currentSessionId = next;
      setSessionIdMock(next);
    },
  ],
}));

interface UIStoreShape {
  sessionToDelete: { id: string; title: string | null | undefined } | null;
  setSessionToDelete: (
    v: { id: string; title: string | null | undefined } | null,
  ) => void;
}

let storeState: UIStoreShape = {
  sessionToDelete: null,
  setSessionToDelete: vi.fn((v) => {
    storeState = { ...storeState, sessionToDelete: v };
  }),
};
vi.mock("../store", () => ({
  useCopilotUIStore: () => storeState,
}));

type DeleteHandler = (vars: { sessionId: string }) => Promise<unknown>;
let deleteImpl: DeleteHandler = async () => ({});
let isPending = false;
const onSuccessRef: { current?: (data: unknown, vars: unknown) => void } = {};
const onErrorRef: { current?: (err: unknown) => void } = {};

vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => {
  return {
    getGetV2ListSessionsQueryKey: () => ["sessions-list"],
    useDeleteV2DeleteSession: ({
      mutation,
    }: {
      mutation?: {
        onSuccess?: (data: unknown, vars: unknown) => void;
        onError?: (err: unknown) => void;
      };
    }) => {
      onSuccessRef.current = mutation?.onSuccess;
      onErrorRef.current = mutation?.onError;
      return {
        mutate: (vars: { sessionId: string }) => {
          deleteImpl(vars).then(
            (data) => onSuccessRef.current?.(data, vars),
            (err) => onErrorRef.current?.(err),
          );
        },
        isPending,
      };
    },
  };
});

import { useSessionDeletion } from "../useSessionDeletion";

function makeWrapper(client: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return React.createElement(QueryClientProvider, { client }, children);
  };
}

describe("useSessionDeletion", () => {
  let client: QueryClient;

  beforeEach(() => {
    mockToast.mockClear();
    setSessionIdMock.mockClear();
    currentSessionId = "s1";
    storeState = {
      sessionToDelete: null,
      setSessionToDelete: vi.fn((v) => {
        storeState = { ...storeState, sessionToDelete: v };
      }),
    };
    deleteImpl = async () => ({});
    isPending = false;
    client = new QueryClient();
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("requestDelete stages a session into the store", () => {
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });
    act(() => {
      result.current.requestDelete("target", "Target chat");
    });
    expect(storeState.setSessionToDelete).toHaveBeenCalledWith({
      id: "target",
      title: "Target chat",
    });
  });

  it("requestDelete is a no-op while a delete is in flight", () => {
    isPending = true;
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });
    act(() => {
      result.current.requestDelete("target", "Target");
    });
    expect(storeState.setSessionToDelete).not.toHaveBeenCalled();
  });

  it("cancelDelete clears the staged session when not deleting", () => {
    storeState = {
      ...storeState,
      sessionToDelete: { id: "target", title: "Target" },
    };
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });
    act(() => {
      result.current.cancelDelete();
    });
    expect(storeState.setSessionToDelete).toHaveBeenCalledWith(null);
  });

  it("cancelDelete is a no-op while a delete is in flight", () => {
    isPending = true;
    storeState = {
      ...storeState,
      sessionToDelete: { id: "target", title: "Target" },
    };
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });
    act(() => {
      result.current.cancelDelete();
    });
    expect(storeState.setSessionToDelete).not.toHaveBeenCalled();
  });

  it("confirmDelete clears the active sessionId on success when it matches", async () => {
    currentSessionId = "s1";
    storeState = {
      ...storeState,
      sessionToDelete: { id: "s1", title: "Active" },
    };
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });

    await act(async () => {
      result.current.confirmDelete();
    });

    expect(setSessionIdMock).toHaveBeenCalledWith(null);
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: ["sessions-list"],
    });
    expect(storeState.setSessionToDelete).toHaveBeenLastCalledWith(null);
  });

  it("confirmDelete keeps the active session when a different one was deleted", async () => {
    currentSessionId = "s1";
    storeState = {
      ...storeState,
      sessionToDelete: { id: "different", title: "Other" },
    };
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });

    await act(async () => {
      result.current.confirmDelete();
    });

    expect(setSessionIdMock).not.toHaveBeenCalled();
    expect(storeState.setSessionToDelete).toHaveBeenLastCalledWith(null);
  });

  it("confirmDelete is a no-op when no session is staged", async () => {
    storeState = { ...storeState, sessionToDelete: null };
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });
    await act(async () => {
      result.current.confirmDelete();
    });
    expect(setSessionIdMock).not.toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("toasts and clears the staged session on delete error", async () => {
    storeState = {
      ...storeState,
      sessionToDelete: { id: "s1", title: "Active" },
    };
    deleteImpl = async () => {
      throw new Error("network down");
    };
    const { result } = renderHook(() => useSessionDeletion(), {
      wrapper: makeWrapper(client),
    });

    await act(async () => {
      result.current.confirmDelete();
    });

    expect(mockToast).toHaveBeenCalledTimes(1);
    expect(mockToast.mock.calls[0][0]).toMatchObject({
      title: "Failed to delete chat",
      variant: "destructive",
    });
    expect(storeState.setSessionToDelete).toHaveBeenLastCalledWith(null);
  });
});
