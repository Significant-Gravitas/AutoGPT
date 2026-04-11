import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { renderHook, act, cleanup } from "@testing-library/react";

// --- Module mocks (must be hoisted before imports) ---

// Bypass useShallow's ref-based shallow comparison so selectors work in tests.
vi.mock("zustand/react/shallow", () => ({
  useShallow: (fn: (s: unknown) => unknown) => fn,
}));

const mockNodes: unknown[] = [];
const mockEdges: unknown[] = [];
const mockSetNodes = vi.fn();
const mockSetEdges = vi.fn();

vi.mock("../../../stores/nodeStore", () => {
  const useNodeStore = (selector: (s: unknown) => unknown) =>
    selector({
      nodes: mockNodes,
      setNodes: mockSetNodes,
    });
  useNodeStore.getState = () => ({
    nodes: mockNodes,
    setNodes: mockSetNodes,
  });
  return { useNodeStore };
});

vi.mock("../../../stores/edgeStore", () => {
  const useEdgeStore = (selector: (s: unknown) => unknown) =>
    selector({
      edges: mockEdges,
      setEdges: mockSetEdges,
    });
  useEdgeStore.getState = () => ({
    edges: mockEdges,
    setEdges: mockSetEdges,
  });
  return { useEdgeStore };
});

const mockPostV2CreateSession = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/chat/chat", () => ({
  postV2CreateSession: (...args: unknown[]) => mockPostV2CreateSession(...args),
}));

vi.mock("@/lib/supabase/actions", () => ({
  getWebSocketToken: vi.fn().mockResolvedValue({ token: "tok", error: null }),
}));

vi.mock("@/services/environment", () => ({
  environment: { getAGPTServerBaseUrl: () => "http://localhost:8000" },
}));

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

const mockSendMessage = vi.fn();
const mockSetMessages = vi.fn();
const mockStop = vi.fn();
let mockChatMessages: unknown[] = [];
let mockChatStatus = "ready";
vi.mock("@ai-sdk/react", () => ({
  useChat: () => ({
    messages: mockChatMessages,
    setMessages: mockSetMessages,
    sendMessage: mockSendMessage,
    stop: mockStop,
    status: mockChatStatus,
    error: undefined,
  }),
}));

vi.mock("ai", () => ({
  // Must be a regular function (not an arrow) so it is constructible via `new`.
  DefaultChatTransport: vi.fn().mockImplementation(function () {
    return {};
  }),
}));

let mockFlowID: string | null = null;
const mockSetQueryStates = vi.fn();

vi.mock("nuqs", () => ({
  parseAsString: { withDefault: (d: string) => d },
  useQueryStates: () => [{ flowID: mockFlowID }, mockSetQueryStates],
}));

// Import after mocks
import {
  useBuilderChatPanel,
  clearGraphSessionCacheForTesting,
} from "../useBuilderChatPanel";

beforeEach(() => {
  mockFlowID = null;
  mockNodes.length = 0;
  mockEdges.length = 0;
  mockChatMessages = [];
  mockChatStatus = "ready";
  mockSetNodes.mockClear();
  mockSetEdges.mockClear();
  mockPostV2CreateSession.mockClear();
  mockSendMessage.mockClear();
  mockSetMessages.mockClear();
  mockToast.mockClear();
  mockSetQueryStates.mockClear();
  clearGraphSessionCacheForTesting();
});

afterEach(() => {
  cleanup();
});

// Flush all pending microtasks + one macrotask so async effects inside `act`
// have time to resolve their awaited promises and commit state updates.
async function openAndFlush(toggle: () => void) {
  await act(async () => {
    toggle();
    await new Promise<void>((resolve) => setTimeout(resolve, 0));
  });
}

describe("useBuilderChatPanel – initial state", () => {
  it("starts with panel closed and no session", () => {
    const { result } = renderHook(() => useBuilderChatPanel());
    expect(result.current.isOpen).toBe(false);
    expect(result.current.sessionId).toBeNull();
    expect(result.current.sessionError).toBe(false);
    expect(result.current.isCreatingSession).toBe(false);
  });

  it("handleToggle opens and closes the panel", () => {
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleToggle();
    });
    expect(result.current.isOpen).toBe(true);

    act(() => {
      result.current.handleToggle();
    });
    expect(result.current.isOpen).toBe(false);
  });
});

describe("useBuilderChatPanel – session lifecycle", () => {
  it("creates session and sets sessionId when panel is opened", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-1" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(mockPostV2CreateSession).toHaveBeenCalledOnce();
    expect(result.current.sessionId).toBe("sess-1");
    expect(result.current.isCreatingSession).toBe(false);
    expect(result.current.sessionError).toBe(false);
  });

  it("sets sessionError when session creation request throws", async () => {
    mockPostV2CreateSession.mockRejectedValue(new Error("network error"));
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(result.current.sessionError).toBe(true);
    expect(result.current.isCreatingSession).toBe(false);
    expect(result.current.sessionId).toBeNull();
  });

  it("sets sessionError when session creation returns non-200 status", async () => {
    mockPostV2CreateSession.mockResolvedValue({ status: 500, data: {} });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(result.current.sessionError).toBe(true);
    expect(result.current.isCreatingSession).toBe(false);
  });

  it("does not create a second session when one already exists", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-existing" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());
    expect(mockPostV2CreateSession).toHaveBeenCalledOnce();

    // Close and reopen — should NOT call postV2CreateSession again
    act(() => result.current.handleToggle());
    await openAndFlush(() => result.current.handleToggle());

    expect(mockPostV2CreateSession).toHaveBeenCalledOnce();
    expect(result.current.sessionId).toBe("sess-existing");
  });

  it("sets sessionError when session creation returns a path-traversal id (security validation)", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "../../admin" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(result.current.sessionError).toBe(true);
    expect(result.current.sessionId).toBeNull();
  });

  it("sets sessionError when session creation returns an id with spaces", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess 1" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(result.current.sessionError).toBe(true);
    expect(result.current.sessionId).toBeNull();
  });
});

describe("useBuilderChatPanel – no auto-send on open", () => {
  it("does NOT auto-send any message when the panel opens", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-open" },
    });
    mockNodes.push({
      id: "n1",
      data: { title: "Search Block", description: "" },
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatPanel – seed message", () => {
  it("sends seed message via sendMessage when session is available and isGraphLoaded=true", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-seed" },
    });
    mockNodes.push({ id: "n1", data: { title: "Search", description: "" } });

    const { result } = renderHook(() =>
      useBuilderChatPanel({ isGraphLoaded: true }),
    );

    await openAndFlush(() => result.current.handleToggle());

    expect(mockSendMessage).toHaveBeenCalledOnce();
    const callArg = mockSendMessage.mock.calls[0][0] as { text: string };
    expect(typeof callArg.text).toBe("string");
    expect(callArg.text).toContain("I'm building an agent");
  });

  it("does NOT send seed message when isGraphLoaded is false (default)", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-no-seed" },
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it("sends seed message only once even when sessionId and isGraphLoaded deps re-run (hasSentSeedMessageRef guard)", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-once" },
    });

    const { result, rerender } = renderHook(() =>
      useBuilderChatPanel({ isGraphLoaded: true }),
    );

    await openAndFlush(() => result.current.handleToggle());
    expect(mockSendMessage).toHaveBeenCalledOnce();

    rerender();

    expect(mockSendMessage).toHaveBeenCalledOnce();
  });

  it("does NOT send seed when panel is closed even if sessionId is cached and isGraphLoaded is true", async () => {
    // Session is pre-cached for this flowID so sessionId is set without opening the panel
    mockFlowID = "flow-cached";
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-cached-pre" },
    });

    // First: open panel to create and cache the session
    const { result, rerender } = renderHook(() =>
      useBuilderChatPanel({ isGraphLoaded: true }),
    );
    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionId).toBe("sess-cached-pre");
    expect(mockSendMessage).toHaveBeenCalledOnce();
    mockSendMessage.mockClear();

    // Close panel, then navigate away and back (clears hasSentSeedMessageRef)
    act(() => result.current.handleToggle()); // close
    mockFlowID = "flow-other";
    rerender();
    mockFlowID = "flow-cached";
    rerender();

    // Panel is still closed but sessionId is restored from cache
    // Seed should NOT fire because panel is closed (nodes = EMPTY_NODES)
    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });
    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatPanel – flowID reset", () => {
  it("resets appliedActionKeys when flowID changes", () => {
    mockNodes.push({ id: "n1", data: { hardcodedValues: {} } });
    mockFlowID = "flow-1";

    const { result, rerender } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n1",
        key: "query",
        value: "test",
      });
    });
    expect(result.current.appliedActionKeys.size).toBe(1);

    mockFlowID = "flow-2";
    rerender();

    expect(result.current.appliedActionKeys.size).toBe(0);
  });

  it("resets sessionId when flowID changes", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-abc" },
    });
    mockFlowID = "flow-1";

    const { result, rerender } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionId).toBe("sess-abc");

    mockFlowID = "flow-2";
    rerender();

    expect(result.current.sessionId).toBeNull();
  });

  it("resets sessionError when flowID changes", async () => {
    mockPostV2CreateSession.mockRejectedValue(new Error("fail"));
    mockFlowID = "flow-1";

    const { result, rerender } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionError).toBe(true);

    mockFlowID = "flow-2";
    rerender();

    expect(result.current.sessionError).toBe(false);
  });

  it("always clears messages on flowID change even when a cached session exists (prevents applied/unapplied mismatch)", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-cached" },
    });
    mockFlowID = "flow-1";

    const { result, rerender } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionId).toBe("sess-cached");

    // Simulate chat messages from the first session
    mockChatMessages = [
      {
        id: "msg-1",
        role: "assistant",
        parts: [{ type: "text", text: "Hello from session 1" }],
      },
    ];
    mockSetMessages.mockClear();

    // Navigate away and back to the same graph — cached session should be restored
    // but messages must be cleared to stay in sync with the reset appliedActionKeys
    mockFlowID = "flow-2";
    rerender();
    mockFlowID = "flow-1";
    rerender();

    // setMessages([]) must be called unconditionally regardless of cached session
    expect(mockSetMessages).toHaveBeenCalledWith([]);
  });
});

describe("useBuilderChatPanel – handleApplyAction", () => {
  it("update_node_input: calls setNodes with merged hardcodedValues (bypasses history)", () => {
    mockNodes.push({
      id: "node-1",
      data: { hardcodedValues: { existing: "value" } },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "node-1",
        key: "query",
        value: "AI news",
      });
    });

    expect(mockSetNodes).toHaveBeenCalledWith([
      {
        id: "node-1",
        data: { hardcodedValues: { existing: "value", query: "AI news" } },
      },
    ]);
  });

  it("update_node_input: shows toast when node not found", () => {
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "nonexistent",
        key: "query",
        value: "test",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("connect_nodes: calls setEdges with new edge appended (bypasses history)", () => {
    mockNodes.push({ id: "src", data: {} }, { id: "tgt", data: {} });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src",
        target: "tgt",
        sourceHandle: "output",
        targetHandle: "input",
      });
    });

    expect(mockSetEdges).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          id: "src:output->tgt:input",
          source: "src",
          target: "tgt",
          sourceHandle: "output",
          targetHandle: "input",
          type: "custom",
          markerEnd: expect.objectContaining({ type: "arrowclosed" }),
        }),
      ]),
    );
  });

  it("connect_nodes: shows toast and does NOT call setEdges when source node is missing", () => {
    mockNodes.push({ id: "tgt", data: {} });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "missing-src",
        target: "tgt",
        sourceHandle: "output",
        targetHandle: "input",
      });
    });

    expect(mockSetEdges).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("connect_nodes: shows toast and does NOT call setEdges when target node is missing", () => {
    mockNodes.push({ id: "src", data: {} });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src",
        target: "missing-tgt",
        sourceHandle: "output",
        targetHandle: "input",
      });
    });

    expect(mockSetEdges).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("update_node_input: rejects key not present in inputSchema", () => {
    mockNodes.push({
      id: "node-1",
      data: {
        hardcodedValues: {},
        inputSchema: { properties: { allowed_key: {} } },
      },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "node-1",
        key: "forbidden_key",
        value: "test",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("update_node_input: allows key present in inputSchema", () => {
    mockNodes.push({
      id: "node-1",
      data: {
        hardcodedValues: {},
        inputSchema: { properties: { query: {} } },
      },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "node-1",
        key: "query",
        value: "AI news",
      });
    });

    expect(mockSetNodes).toHaveBeenCalledWith([
      {
        id: "node-1",
        data: {
          hardcodedValues: { query: "AI news" },
          inputSchema: { properties: { query: {} } },
        },
      },
    ]);
  });

  it("connect_nodes: rejects sourceHandle not in outputSchema", () => {
    mockNodes.push(
      { id: "src", data: { outputSchema: { properties: { result: {} } } } },
      { id: "tgt", data: { inputSchema: { properties: { input: {} } } } },
    );
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src",
        target: "tgt",
        sourceHandle: "nonexistent_output",
        targetHandle: "input",
      });
    });

    expect(mockSetEdges).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("connect_nodes: rejects targetHandle not in inputSchema", () => {
    mockNodes.push(
      { id: "src", data: { outputSchema: { properties: { result: {} } } } },
      { id: "tgt", data: { inputSchema: { properties: { input: {} } } } },
    );
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src",
        target: "tgt",
        sourceHandle: "result",
        targetHandle: "nonexistent_input",
      });
    });

    expect(mockSetEdges).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("connect_nodes: calls setEdges when both handles are valid according to schemas", () => {
    mockNodes.push(
      { id: "src", data: { outputSchema: { properties: { result: {} } } } },
      { id: "tgt", data: { inputSchema: { properties: { input: {} } } } },
    );
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src",
        target: "tgt",
        sourceHandle: "result",
        targetHandle: "input",
      });
    });

    expect(mockSetEdges).toHaveBeenCalledWith(
      expect.arrayContaining([
        expect.objectContaining({
          id: "src:result->tgt:input",
          source: "src",
          target: "tgt",
          sourceHandle: "result",
          targetHandle: "input",
          type: "custom",
          markerEnd: expect.objectContaining({ type: "arrowclosed" }),
        }),
      ]),
    );
  });

  it("adds action key to appliedActionKeys after successful apply", () => {
    mockNodes.push({ id: "n1", data: { hardcodedValues: {} } });
    const { result } = renderHook(() => useBuilderChatPanel());

    const action = {
      type: "update_node_input" as const,
      nodeId: "n1",
      key: "query",
      value: "test",
    };

    act(() => {
      result.current.handleApplyAction(action);
    });

    expect(result.current.appliedActionKeys.has('n1:query:"test"')).toBe(true);
  });
});

describe("useBuilderChatPanel – undo", () => {
  it("restores previous node state after undo using setNodes (bypasses history store)", () => {
    const initialNode = {
      id: "node-undo",
      data: { hardcodedValues: { existing: "original" } },
    };
    mockNodes.push(initialNode);

    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "node-undo",
        key: "query",
        value: "changed",
      });
    });

    expect(result.current.undoStack).toHaveLength(1);

    // Clear call history so we can verify undo uses setNodes with the original snapshot
    mockSetNodes.mockClear();

    act(() => {
      result.current.handleUndoLastAction();
    });

    // setNodes is called with the captured snapshot to bypass the global history store
    expect(mockSetNodes).toHaveBeenCalledWith([initialNode]);
    expect(result.current.undoStack).toHaveLength(0);
  });

  it("removes action key from appliedActionKeys after undo", () => {
    mockNodes.push({ id: "n-undo", data: { hardcodedValues: {} } });

    const { result } = renderHook(() => useBuilderChatPanel());

    const action = {
      type: "update_node_input" as const,
      nodeId: "n-undo",
      key: "val",
      value: "x",
    };

    act(() => {
      result.current.handleApplyAction(action);
    });
    expect(result.current.appliedActionKeys.size).toBe(1);

    act(() => {
      result.current.handleUndoLastAction();
    });
    expect(result.current.appliedActionKeys.size).toBe(0);
  });

  it("connect_nodes: restores edges via setEdges after undo (bypasses history store)", () => {
    const initialEdge = { id: "existing-edge", source: "a", target: "b" };
    mockEdges.push(initialEdge);
    mockNodes.push({ id: "src", data: {} }, { id: "tgt", data: {} });

    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src",
        target: "tgt",
        sourceHandle: "out",
        targetHandle: "in",
      });
    });

    expect(mockSetEdges).toHaveBeenCalledOnce();
    expect(result.current.undoStack).toHaveLength(1);

    mockSetEdges.mockClear();

    act(() => {
      result.current.handleUndoLastAction();
    });

    // setEdges is called with the original captured snapshot to bypass the global history store
    expect(mockSetEdges).toHaveBeenCalledWith([initialEdge]);
    expect(result.current.undoStack).toHaveLength(0);
    expect(result.current.appliedActionKeys.size).toBe(0);
  });
});

describe("useBuilderChatPanel – parsedActions integration", () => {
  it("returns parsed actions from assistant messages when status is ready", () => {
    mockChatMessages = [
      {
        id: "msg-1",
        role: "assistant",
        parts: [
          {
            type: "text",
            text: '```json\n{"action":"update_node_input","node_id":"n1","key":"query","value":"AI news"}\n```',
          },
        ],
      },
    ];
    mockChatStatus = "ready";

    const { result } = renderHook(() => useBuilderChatPanel());

    expect(result.current.parsedActions).toHaveLength(1);
    expect(result.current.parsedActions[0]).toEqual({
      type: "update_node_input",
      nodeId: "n1",
      key: "query",
      value: "AI news",
    });
  });

  it("returns empty parsedActions when status is streaming", () => {
    mockChatMessages = [
      {
        id: "msg-1",
        role: "assistant",
        parts: [
          {
            type: "text",
            text: '```json\n{"action":"update_node_input","node_id":"n1","key":"query","value":"AI news"}\n```',
          },
        ],
      },
    ];
    mockChatStatus = "streaming";

    const { result } = renderHook(() => useBuilderChatPanel());

    expect(result.current.parsedActions).toHaveLength(0);
  });

  it("deduplicates identical actions from multiple assistant messages", () => {
    const actionBlock =
      '```json\n{"action":"update_node_input","node_id":"n1","key":"query","value":"AI news"}\n```';
    mockChatMessages = [
      {
        id: "msg-1",
        role: "assistant",
        parts: [{ type: "text", text: actionBlock }],
      },
      {
        id: "msg-2",
        role: "assistant",
        parts: [{ type: "text", text: actionBlock }],
      },
    ];
    mockChatStatus = "ready";

    const { result } = renderHook(() => useBuilderChatPanel());

    expect(result.current.parsedActions).toHaveLength(1);
  });

  it("does NOT re-parse stale messages from the previous graph after navigation (sentry race PRRT_kwDOJKSTjM56RVeU)", () => {
    // Reproduces the navigation race: when flowID changes, the cleanup
    // effect resets the parsed-actions cache and queues setMessages([]),
    // but the parse-actions effect runs in the same effect cycle while
    // the messages closure still holds the previous graph's messages.
    // Without the navigation guard, the parser would re-scan those stale
    // messages from index 0 (because the cache was reset) and populate
    // parsedActions with the previous graph's actions.
    const flow1Action =
      '```json\n{"action":"update_node_input","node_id":"flow-1-node","key":"query","value":"flow-1 value"}\n```';
    mockChatMessages = [
      {
        id: "msg-1",
        role: "assistant",
        parts: [{ type: "text", text: flow1Action }],
      },
    ];
    mockChatStatus = "ready";
    mockFlowID = "flow-1";

    const { result, rerender } = renderHook(() => useBuilderChatPanel());

    // Initial mount on flow-1: actions parsed normally.
    expect(result.current.parsedActions).toHaveLength(1);
    expect(result.current.parsedActions[0]).toMatchObject({
      nodeId: "flow-1-node",
    });

    // Simulate navigation to flow-2. The test mock keeps `mockChatMessages`
    // pointing at the flow-1 messages (mirroring the real race window where
    // useChat hasn't yet picked up `setMessages([])`).
    mockFlowID = "flow-2";
    rerender();

    // The navigation guard must prevent the parse-actions effect from
    // re-populating parsedActions with the stale flow-1 message it can
    // still see in its closure.
    expect(result.current.parsedActions).toHaveLength(0);
  });
});

describe("useBuilderChatPanel – Escape key handler", () => {
  it("closes the panel when Escape is pressed while open", () => {
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleToggle();
    });
    expect(result.current.isOpen).toBe(true);

    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });
    expect(result.current.isOpen).toBe(false);
  });

  it("does not error when Escape is pressed while panel is closed", () => {
    const { result } = renderHook(() => useBuilderChatPanel());
    expect(result.current.isOpen).toBe(false);

    act(() => {
      document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape" }));
    });

    expect(result.current.isOpen).toBe(false);
  });
});

describe("useBuilderChatPanel – retrySession", () => {
  it("clears sessionError so the session-creation effect can re-run", async () => {
    mockPostV2CreateSession.mockRejectedValueOnce(new Error("network error"));

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionError).toBe(true);

    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-retry" },
    });

    await act(async () => {
      result.current.retrySession();
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    });

    expect(result.current.sessionError).toBe(false);
    expect(result.current.sessionId).toBe("sess-retry");
  });

  it("re-sends seed message to new session after retry (hasSentSeedMessageRef is reset)", async () => {
    // First session succeeds and seed is sent
    mockPostV2CreateSession.mockResolvedValueOnce({
      status: 200,
      data: { id: "sess-first" },
    });
    const { result } = renderHook(() =>
      useBuilderChatPanel({ isGraphLoaded: true }),
    );
    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionId).toBe("sess-first");
    expect(mockSendMessage).toHaveBeenCalledOnce();

    // Force a retry: evict cache and set error state manually, then retry
    mockSendMessage.mockClear();
    mockPostV2CreateSession.mockResolvedValueOnce({
      status: 200,
      data: { id: "sess-retry-seed" },
    });
    await act(async () => {
      result.current.retrySession();
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    });

    // New session obtained; seed message must be sent again to the new session
    expect(result.current.sessionId).toBe("sess-retry-seed");
    expect(mockSendMessage).toHaveBeenCalledOnce();
  });

  it("clears stale messages when retrySession is called (setMessages reset)", async () => {
    // Simulate stale messages from a previous session
    mockChatMessages = [
      {
        id: "stale-1",
        role: "assistant",
        parts: [{ type: "text", text: "Old message from failed session" }],
      },
    ];

    const { result } = renderHook(() => useBuilderChatPanel());

    // Messages should be present before retry (from mock)
    expect(result.current.messages).toHaveLength(1);

    act(() => {
      result.current.retrySession();
    });

    // setMessages([]) clears the internal useChat message list
    expect(mockSetMessages).toHaveBeenCalledWith([]);
  });
});

describe("useBuilderChatPanel – handleSend", () => {
  it("clears inputValue after sending when session is ready", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-send" },
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    act(() => {
      result.current.setInputValue("hello world");
    });

    act(() => {
      result.current.handleSend();
    });

    expect(result.current.inputValue).toBe("");
    expect(mockSendMessage).toHaveBeenCalledWith({ text: "hello world" });
  });

  it("does not send when inputValue is whitespace only", () => {
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.setInputValue("   ");
    });

    act(() => {
      result.current.handleSend();
    });

    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it("does not send when canSend is false (sessionError=true)", async () => {
    mockPostV2CreateSession.mockRejectedValue(new Error("fail"));
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());
    expect(result.current.sessionError).toBe(true);
    expect(result.current.canSend).toBe(false);

    act(() => {
      result.current.setInputValue("hello");
    });

    act(() => {
      result.current.handleSend();
    });

    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatPanel – handleKeyDown", () => {
  it("calls handleSend on Enter without Shift when canSend is true", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-kd" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    act(() => {
      result.current.setInputValue("test message");
    });

    const mockPreventDefault = vi.fn();
    act(() => {
      result.current.handleKeyDown({
        key: "Enter",
        shiftKey: false,
        preventDefault: mockPreventDefault,
      } as unknown as import("react").KeyboardEvent<HTMLTextAreaElement>);
    });

    expect(mockPreventDefault).toHaveBeenCalled();
    expect(mockSendMessage).toHaveBeenCalledWith({ text: "test message" });
  });

  it("does NOT call handleSend on Shift+Enter (allows newline insertion)", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-shift" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    act(() => {
      result.current.setInputValue("multiline");
    });

    const mockPreventDefault = vi.fn();
    act(() => {
      result.current.handleKeyDown({
        key: "Enter",
        shiftKey: true,
        preventDefault: mockPreventDefault,
      } as unknown as import("react").KeyboardEvent<HTMLTextAreaElement>);
    });

    expect(mockPreventDefault).not.toHaveBeenCalled();
    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatPanel – schema-absent nodes", () => {
  it("update_node_input: allows any key when node has no inputSchema (permissive mode)", () => {
    mockNodes.push({
      id: "schema-less",
      data: { hardcodedValues: {} },
      // No inputSchema at all
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "schema-less",
        key: "any_key",
        value: "any_value",
      });
    });

    // Without a schema, validation is skipped — the key is applied permissively
    expect(mockSetNodes).toHaveBeenCalledWith([
      {
        id: "schema-less",
        data: { hardcodedValues: { any_key: "any_value" } },
      },
    ]);
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("connect_nodes: allows connection when source node has no outputSchema (permissive mode)", () => {
    mockNodes.push(
      { id: "src-no-schema", data: {} }, // no outputSchema
      {
        id: "tgt-has-schema",
        data: { inputSchema: { properties: { input: {} } } },
      },
    );
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src-no-schema",
        target: "tgt-has-schema",
        sourceHandle: "any_output",
        targetHandle: "input",
      });
    });

    // Without an outputSchema, sourceHandle validation is skipped
    expect(mockSetEdges).toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();
  });

  it("connect_nodes: allows connection when target node has no inputSchema (permissive mode)", () => {
    mockNodes.push(
      {
        id: "src-has-schema",
        data: { outputSchema: { properties: { output: {} } } },
      },
      { id: "tgt-no-schema", data: {} }, // no inputSchema
    );
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "connect_nodes",
        source: "src-has-schema",
        target: "tgt-no-schema",
        sourceHandle: "output",
        targetHandle: "any_input",
      });
    });

    // Without an inputSchema, targetHandle validation is skipped
    expect(mockSetEdges).toHaveBeenCalled();
    expect(mockToast).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatPanel – sequential multi-undo (LIFO order)", () => {
  it("undoes two applied actions in LIFO order, restoring correct state at each step", () => {
    const initialNode = {
      id: "n1",
      data: { hardcodedValues: { x: "original" } },
    };
    mockNodes.push(initialNode);

    const { result } = renderHook(() => useBuilderChatPanel());

    // Apply first action
    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n1",
        key: "x",
        value: "first_change",
      });
    });
    expect(result.current.undoStack).toHaveLength(1);

    // Apply second action
    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n1",
        key: "x",
        value: "second_change",
      });
    });
    expect(result.current.undoStack).toHaveLength(2);

    // Undo second action — should restore to snapshot taken before second action
    // (which captured the state after first action, i.e. mockNodes at that point)
    mockSetNodes.mockClear();
    act(() => {
      result.current.handleUndoLastAction();
    });
    expect(result.current.undoStack).toHaveLength(1);
    // setNodes called with the snapshot captured before second action applied
    expect(mockSetNodes).toHaveBeenCalledOnce();

    // Undo first action — should restore to snapshot taken before first action
    mockSetNodes.mockClear();
    act(() => {
      result.current.handleUndoLastAction();
    });
    expect(result.current.undoStack).toHaveLength(0);
    expect(mockSetNodes).toHaveBeenCalledWith([initialNode]);
  });
});

describe("useBuilderChatPanel – duplicate edge guard", () => {
  it("does not append duplicate edge when same connect_nodes action is applied twice", () => {
    mockNodes.push({ id: "src", data: {} }, { id: "tgt", data: {} });

    const action = {
      type: "connect_nodes" as const,
      source: "src",
      target: "tgt",
      sourceHandle: "out",
      targetHandle: "in",
    };

    // Simulate the edge store updating when setEdges is called
    const newEdge = {
      id: "src:out->tgt:in",
      source: "src",
      target: "tgt",
      sourceHandle: "out",
      targetHandle: "in",
      type: "custom",
    };
    mockSetEdges.mockImplementationOnce((edges: unknown[]) => {
      mockEdges.push(...edges);
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction(action);
    });

    expect(mockSetEdges).toHaveBeenCalledOnce();
    expect(result.current.appliedActionKeys.size).toBe(1);
    // Verify the edge is now in the mock store
    expect(mockEdges).toContainEqual(expect.objectContaining(newEdge));

    // Second apply of the same action — should not call setEdges again
    mockSetEdges.mockClear();
    act(() => {
      result.current.handleApplyAction(action);
    });

    // setEdges should NOT be called again — the edge already exists in the store
    expect(mockSetEdges).not.toHaveBeenCalled();
    // But appliedActionKeys should still contain the key
    expect(result.current.appliedActionKeys.size).toBe(1);
  });
});

describe("useBuilderChatPanel – undo stack size cap", () => {
  it("caps the undo stack at MAX_UNDO (20) entries, dropping the oldest", () => {
    // Push 21 nodes so each apply action targets a unique node
    for (let i = 0; i <= 20; i++) {
      mockNodes.push({ id: `n${i}`, data: { hardcodedValues: {} } });
    }

    const { result } = renderHook(() => useBuilderChatPanel());

    // Apply 21 actions
    for (let i = 0; i <= 20; i++) {
      act(() => {
        result.current.handleApplyAction({
          type: "update_node_input",
          nodeId: `n${i}`,
          key: "v",
          value: `val${i}`,
        });
      });
    }

    // Stack should be capped at 20
    expect(result.current.undoStack).toHaveLength(20);
  });
});

describe("useBuilderChatPanel – handleUndoLastAction on empty stack", () => {
  it("does nothing when undoStack is empty", () => {
    const { result } = renderHook(() => useBuilderChatPanel());

    expect(result.current.undoStack).toHaveLength(0);

    // Should not throw or call setNodes/setEdges
    act(() => {
      result.current.handleUndoLastAction();
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockSetEdges).not.toHaveBeenCalled();
    expect(result.current.undoStack).toHaveLength(0);
  });
});

describe("useBuilderChatPanel – transport prepareSendMessagesRequest", () => {
  it("calls getWebSocketToken and returns correct request body", async () => {
    const { getWebSocketToken } = await import("@/lib/supabase/actions");
    const { DefaultChatTransport } = await import("ai");
    const MockTransport = DefaultChatTransport as ReturnType<typeof vi.fn>;

    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-transport" },
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    expect(MockTransport).toHaveBeenCalled();
    const ctorArg = MockTransport.mock.calls[
      MockTransport.mock.calls.length - 1
    ][0] as {
      prepareSendMessagesRequest: (args: {
        messages: unknown[];
      }) => Promise<unknown>;
    };
    expect(typeof ctorArg.prepareSendMessagesRequest).toBe("function");

    const messages = [
      { role: "user", parts: [{ type: "text", text: "hello" }] },
    ];
    const req = await ctorArg.prepareSendMessagesRequest({ messages });

    expect(getWebSocketToken).toHaveBeenCalled();
    expect(req).toMatchObject({
      body: { message: "hello", is_user_message: true },
      headers: { Authorization: "Bearer tok" },
    });
  });

  it("throws when getWebSocketToken returns null token", async () => {
    const { getWebSocketToken } = await import("@/lib/supabase/actions");
    const { DefaultChatTransport } = await import("ai");
    const MockTransport = DefaultChatTransport as ReturnType<typeof vi.fn>;

    vi.mocked(getWebSocketToken).mockResolvedValueOnce({
      token: null,
      error: "auth failed",
    });

    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-auth-fail" },
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    const ctorArg = MockTransport.mock.calls[
      MockTransport.mock.calls.length - 1
    ][0] as {
      prepareSendMessagesRequest: (args: {
        messages: unknown[];
      }) => Promise<unknown>;
    };
    const messages = [{ role: "user", parts: [{ type: "text", text: "hi" }] }];
    await expect(
      ctorArg.prepareSendMessagesRequest({ messages }),
    ).rejects.toThrow("Authentication failed");
  });

  it("throws when messages array is empty (empty messages guard)", async () => {
    const { DefaultChatTransport } = await import("ai");
    const MockTransport = DefaultChatTransport as ReturnType<typeof vi.fn>;

    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-empty-msg" },
    });

    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    const ctorArg = MockTransport.mock.calls[
      MockTransport.mock.calls.length - 1
    ][0] as {
      prepareSendMessagesRequest: (args: {
        messages: unknown[];
      }) => Promise<unknown>;
    };
    await expect(
      ctorArg.prepareSendMessagesRequest({ messages: [] }),
    ).rejects.toThrow("No message to send");
  });
});

describe("useBuilderChatPanel – handleKeyDown empty input guard", () => {
  it("does NOT call sendMessage on Enter when inputValue is empty", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-empty" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    await openAndFlush(() => result.current.handleToggle());

    const mockPreventDefault = vi.fn();
    act(() => {
      result.current.handleKeyDown({
        key: "Enter",
        shiftKey: false,
        preventDefault: mockPreventDefault,
      } as unknown as import("react").KeyboardEvent<HTMLTextAreaElement>);
    });

    expect(mockSendMessage).not.toHaveBeenCalled();
  });
});

describe("useBuilderChatPanel – inputValue resets on flowID change", () => {
  it("clears inputValue when flowID changes", () => {
    mockFlowID = "flow-a";
    const { result, rerender } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.setInputValue("typed text");
    });
    expect(result.current.inputValue).toBe("typed text");

    mockFlowID = "flow-b";
    rerender();

    expect(result.current.inputValue).toBe("");
  });
});

describe("useBuilderChatPanel – prototype pollution guard", () => {
  it("rejects __proto__ as a key when node has an inputSchema with properties", () => {
    mockNodes.push({
      id: "n-proto",
      data: {
        hardcodedValues: {},
        inputSchema: { properties: { query: {} } },
      },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    const protoBefore = Object.prototype.hasOwnProperty("injected");

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n-proto",
        key: "__proto__",
        value: "injected",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
    expect(Object.prototype.hasOwnProperty("injected")).toBe(protoBefore);
  });

  it("rejects constructor as a key when node has an inputSchema with properties", () => {
    mockNodes.push({
      id: "n-ctor",
      data: {
        hardcodedValues: {},
        inputSchema: { properties: { query: {} } },
      },
    });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n-ctor",
        key: "constructor",
        value: "injected",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });
});

describe("useBuilderChatPanel – tool call detection", () => {
  function makeDynamicToolPart(
    toolName: string,
    toolCallId: string,
    state: string,
    output: unknown = null,
  ) {
    return { type: "dynamic-tool", toolName, toolCallId, state, output };
  }

  it("calls onGraphEdited when edit_agent tool call completes", async () => {
    mockChatStatus = "ready";
    mockChatMessages = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          makeDynamicToolPart("edit_agent", "tc-1", "output-available", null),
        ],
      },
    ];
    const onGraphEdited = vi.fn();
    renderHook(() => useBuilderChatPanel({ onGraphEdited }));

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(onGraphEdited).toHaveBeenCalledOnce();
  });

  it("does NOT call onGraphEdited for a tool call that is not output-available", async () => {
    mockChatStatus = "ready";
    mockChatMessages = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          makeDynamicToolPart("edit_agent", "tc-pending", "pending", null),
        ],
      },
    ];
    const onGraphEdited = vi.fn();
    renderHook(() => useBuilderChatPanel({ onGraphEdited }));

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(onGraphEdited).not.toHaveBeenCalled();
  });

  it("does NOT call onGraphEdited when status is streaming", async () => {
    mockChatStatus = "streaming";
    mockChatMessages = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          makeDynamicToolPart(
            "edit_agent",
            "tc-stream",
            "output-available",
            null,
          ),
        ],
      },
    ];
    const onGraphEdited = vi.fn();
    renderHook(() => useBuilderChatPanel({ onGraphEdited }));

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(onGraphEdited).not.toHaveBeenCalled();
  });

  it("does NOT process the same tool call twice (processedToolCallsRef deduplication)", async () => {
    mockChatStatus = "ready";
    const part = makeDynamicToolPart(
      "edit_agent",
      "tc-dedup",
      "output-available",
      null,
    );
    mockChatMessages = [{ id: "m1", role: "assistant", parts: [part] }];

    const onGraphEdited = vi.fn();
    const { rerender } = renderHook(() =>
      useBuilderChatPanel({ onGraphEdited }),
    );

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(onGraphEdited).toHaveBeenCalledOnce();

    act(() => rerender());

    expect(onGraphEdited).toHaveBeenCalledOnce();
  });

  it("calls setQueryStates with execution_id when run_agent tool call completes with valid id", async () => {
    mockChatStatus = "ready";
    mockChatMessages = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          makeDynamicToolPart("run_agent", "tc-run", "output-available", {
            execution_id: "exec-abc123",
          }),
        ],
      },
    ];
    renderHook(() => useBuilderChatPanel());

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(mockSetQueryStates).toHaveBeenCalledWith({
      flowExecutionID: "exec-abc123",
    });
  });

  it("does NOT call setQueryStates when run_agent output has no execution_id", async () => {
    mockChatStatus = "ready";
    mockChatMessages = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          makeDynamicToolPart("run_agent", "tc-run-null", "output-available", {
            other_field: "something",
          }),
        ],
      },
    ];
    renderHook(() => useBuilderChatPanel());

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(mockSetQueryStates).not.toHaveBeenCalled();
  });

  it("does NOT call setQueryStates when run_agent execution_id contains path-traversal characters", async () => {
    mockChatStatus = "ready";
    mockChatMessages = [
      {
        id: "m1",
        role: "assistant",
        parts: [
          makeDynamicToolPart("run_agent", "tc-run-bad", "output-available", {
            execution_id: "../../admin",
          }),
        ],
      },
    ];
    renderHook(() => useBuilderChatPanel());

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(mockSetQueryStates).not.toHaveBeenCalled();
  });

  it("does NOT process run_agent tool call twice (deduplication)", async () => {
    mockChatStatus = "ready";
    const part = makeDynamicToolPart(
      "run_agent",
      "tc-run-dedup",
      "output-available",
      {
        execution_id: "exec-dedup",
      },
    );
    mockChatMessages = [{ id: "m1", role: "assistant", parts: [part] }];

    const { rerender } = renderHook(() => useBuilderChatPanel());

    await act(async () => {
      await new Promise<void>((r) => setTimeout(r, 0));
    });

    expect(mockSetQueryStates).toHaveBeenCalledOnce();

    act(() => rerender());

    expect(mockSetQueryStates).toHaveBeenCalledOnce();
  });
});

describe("useBuilderChatPanel – prototype pollution blocklist (no-schema nodes)", () => {
  it("rejects __proto__ even when node has no inputSchema", () => {
    mockNodes.push({ id: "n-schema-less", data: { hardcodedValues: {} } });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n-schema-less",
        key: "__proto__",
        value: "injected",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("rejects constructor even when node has no inputSchema", () => {
    mockNodes.push({ id: "n-ctor-no-schema", data: { hardcodedValues: {} } });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n-ctor-no-schema",
        key: "constructor",
        value: "injected",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });

  it("rejects prototype even when node has no inputSchema", () => {
    mockNodes.push({ id: "n-proto-no-schema", data: { hardcodedValues: {} } });
    const { result } = renderHook(() => useBuilderChatPanel());

    act(() => {
      result.current.handleApplyAction({
        type: "update_node_input",
        nodeId: "n-proto-no-schema",
        key: "prototype",
        value: "injected",
      });
    });

    expect(mockSetNodes).not.toHaveBeenCalled();
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({ variant: "destructive" }),
    );
  });
});

describe("useBuilderChatPanel – sendRawMessage length clamp", () => {
  async function openPanel(result: { current: { handleToggle: () => void } }) {
    await openAndFlush(() => result.current.handleToggle());
  }

  it("drops empty input without calling sendMessage", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-1" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());
    await openPanel(result);

    act(() => {
      result.current.sendRawMessage("");
    });
    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it("does not send when canSend is false (no session yet)", () => {
    const { result } = renderHook(() => useBuilderChatPanel());
    act(() => {
      result.current.sendRawMessage("hello");
    });
    expect(mockSendMessage).not.toHaveBeenCalled();
  });

  it("forwards text as-is when under the 4000-char cap", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-2" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());
    await openPanel(result);

    const msg = "hello world";
    act(() => {
      result.current.sendRawMessage(msg);
    });
    expect(mockSendMessage).toHaveBeenCalledWith({ text: msg });
  });

  it("truncates input longer than 4000 characters", async () => {
    mockPostV2CreateSession.mockResolvedValue({
      status: 200,
      data: { id: "sess-3" },
    });
    const { result } = renderHook(() => useBuilderChatPanel());
    await openPanel(result);

    const huge = "a".repeat(5000);
    act(() => {
      result.current.sendRawMessage(huge);
    });
    expect(mockSendMessage).toHaveBeenCalledTimes(1);
    const sentText = (mockSendMessage.mock.calls[0][0] as { text: string })
      .text;
    expect(sentText.length).toBe(4000);
    expect(sentText).toBe("a".repeat(4000));
  });
});
