import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";
import { BuilderChatPanel } from "../BuilderChatPanel";
import {
  serializeGraphForChat,
  parseGraphActions,
  getActionKey,
  getNodeDisplayName,
  buildSeedPrompt,
  extractTextFromParts,
  SEED_PROMPT_PREFIX,
} from "../helpers";
import type { CustomNode } from "../../FlowEditor/nodes/CustomNode/CustomNode";
import type { CustomEdge } from "../../FlowEditor/edges/CustomEdge";

// Mock the hook so we isolate the component rendering
vi.mock("../useBuilderChatPanel", () => ({
  useBuilderChatPanel: vi.fn(),
}));

import { useBuilderChatPanel } from "../useBuilderChatPanel";

const mockUseBuilderChatPanel = vi.mocked(useBuilderChatPanel);

function makeMockHook(
  overrides: Partial<ReturnType<typeof useBuilderChatPanel>> = {},
): ReturnType<typeof useBuilderChatPanel> {
  return {
    isOpen: false,
    handleToggle: vi.fn(),
    retrySession: vi.fn(),
    messages: [],
    stop: vi.fn(),
    error: undefined,
    isCreatingSession: false,
    sessionError: false,
    sessionId: null,
    nodes: [],
    parsedActions: [],
    appliedActionKeys: new Set<string>(),
    handleApplyAction: vi.fn(),
    undoStack: [],
    handleUndoLastAction: vi.fn(),
    inputValue: "",
    setInputValue: vi.fn(),
    handleSend: vi.fn(),
    sendRawMessage: vi.fn(),
    handleKeyDown: vi.fn(),
    isStreaming: false,
    canSend: false,
    ...overrides,
  };
}

beforeEach(() => {
  mockUseBuilderChatPanel.mockReturnValue(makeMockHook());
});

afterEach(() => {
  cleanup();
});

describe("BuilderChatPanel", () => {
  it("renders the toggle button when closed", () => {
    render(<BuilderChatPanel />);
    expect(screen.getByLabelText("Chat with builder")).toBeDefined();
  });

  it("does not render the panel content when closed", () => {
    render(<BuilderChatPanel />);
    expect(screen.queryByText("Chat with Builder")).toBeNull();
  });

  it("calls handleToggle when the toggle button is clicked", () => {
    const handleToggle = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(makeMockHook({ handleToggle }));
    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByLabelText("Chat with builder"));
    expect(handleToggle).toHaveBeenCalledOnce();
  });

  it("renders the panel when isOpen is true", () => {
    mockUseBuilderChatPanel.mockReturnValue(makeMockHook({ isOpen: true }));
    render(<BuilderChatPanel />);
    expect(screen.getByText("Chat with Builder")).toBeDefined();
  });

  it("shows creating session indicator when isCreatingSession is true", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, isCreatingSession: true }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText(/Setting up chat session/i)).toBeDefined();
  });

  it("shows welcome/empty state when there are no messages", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, messages: [] }),
    );
    render(<BuilderChatPanel />);
    expect(
      screen.getByText(/Ask me to explain or modify your agent/i),
    ).toBeDefined();
  });

  it("renders user and assistant messages", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        messages: [
          {
            id: "1",
            role: "user",
            parts: [{ type: "text", text: "What does this agent do?" }],
          },
          {
            id: "2",
            role: "assistant",
            parts: [{ type: "text", text: "This agent searches the web." }],
          },
        ] as ReturnType<typeof useBuilderChatPanel>["messages"],
      }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText("What does this agent do?")).toBeDefined();
    expect(screen.getByText("This agent searches the web.")).toBeDefined();
  });

  it("renders suggested changes section when parsedActions are present", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        parsedActions: [
          {
            type: "update_node_input",
            nodeId: "1",
            key: "query",
            value: "AI news",
          },
        ],
      }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText("Suggested changes")).toBeDefined();
  });

  it("renders the action label correctly for update_node_input", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "Search",
          description: "",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
    ] as unknown as CustomNode[];

    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        nodes,
        parsedActions: [
          {
            type: "update_node_input",
            nodeId: "1",
            key: "query",
            value: "AI news",
          },
        ],
      }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText(`Set "Search" "query" = "AI news"`)).toBeDefined();
  });

  it("shows Apply button for unapplied actions and Applied badge for applied actions", () => {
    const action = {
      type: "update_node_input" as const,
      nodeId: "1",
      key: "query",
      value: "AI news",
    };
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        parsedActions: [action],
        appliedActionKeys: new Set([getActionKey(action)]),
      }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText("Applied")).toBeDefined();
    expect(screen.queryByText("Apply")).toBeNull();
  });

  it("calls handleApplyAction when Apply button is clicked", () => {
    const handleApplyAction = vi.fn();
    const action = {
      type: "update_node_input" as const,
      nodeId: "1",
      key: "query",
      value: "AI news",
    };
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        parsedActions: [action],
        handleApplyAction,
      }),
    );
    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByText("Apply"));
    expect(handleApplyAction).toHaveBeenCalledWith(action);
  });

  it("does not call handleSend when the textarea is empty and Send button is disabled", () => {
    const handleSend = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        sessionId: "sess-1",
        canSend: true,
        inputValue: "",
        handleSend,
      }),
    );
    render(<BuilderChatPanel />);
    const sendButton = screen.getByLabelText("Send");
    expect((sendButton as HTMLButtonElement).disabled).toBe(true);
    fireEvent.click(sendButton);
    expect(handleSend).not.toHaveBeenCalled();
  });

  it("calls handleSend when the Send button is clicked with text", () => {
    const handleSend = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        sessionId: "sess-1",
        canSend: true,
        inputValue: "Add a summarizer block",
        handleSend,
      }),
    );
    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByLabelText("Send"));
    expect(handleSend).toHaveBeenCalledOnce();
  });

  it("calls handleKeyDown when a key is pressed in the textarea", () => {
    const handleKeyDown = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        sessionId: "sess-1",
        canSend: true,
        inputValue: "Explain this agent",
        handleKeyDown,
      }),
    );
    render(<BuilderChatPanel />);
    const textarea = screen.getByPlaceholderText(/Ask about your agent/i);
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(handleKeyDown).toHaveBeenCalled();
  });

  it("shows Stop button when streaming", () => {
    const stop = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, isStreaming: true, stop }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByLabelText("Stop")).toBeDefined();
    fireEvent.click(screen.getByLabelText("Stop"));
    expect(stop).toHaveBeenCalledOnce();
  });

  it("shows stream error when error prop is set", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        error: new Error("Connection failed"),
      }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText(/Connection error/i)).toBeDefined();
  });

  it("shows session error message with Retry when sessionError is true", () => {
    const retrySession = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, sessionError: true, retrySession }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText(/Failed to start chat session/i)).toBeDefined();
    expect(screen.getByText("Retry")).toBeDefined();
    fireEvent.click(screen.getByText("Retry"));
    expect(retrySession).toHaveBeenCalledOnce();
  });

  it("renders the panel with role=complementary and message list with role=log", () => {
    mockUseBuilderChatPanel.mockReturnValue(makeMockHook({ isOpen: true }));
    render(<BuilderChatPanel />);
    expect(screen.getByRole("complementary")).toBeDefined();
    expect(screen.getByRole("log")).toBeDefined();
  });

  it("shows undo button in header when undoStack has entries", () => {
    const handleUndoLastAction = vi.fn();
    const fakeRestore = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        undoStack: [{ actionKey: "n1:query", restore: fakeRestore }],
        handleUndoLastAction,
      }),
    );
    render(<BuilderChatPanel />);
    const undoBtn = screen.getByLabelText("Undo last applied change");
    expect(undoBtn).toBeDefined();
    fireEvent.click(undoBtn);
    expect(handleUndoLastAction).toHaveBeenCalledOnce();
  });

  it("does not show undo button when undoStack is empty", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, undoStack: [] }),
    );
    render(<BuilderChatPanel />);
    expect(screen.queryByLabelText("Undo last applied change")).toBeNull();
  });

  it("hides the seed message from the chat UI", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({
        isOpen: true,
        messages: [
          {
            id: "seed",
            role: "user",
            parts: [
              {
                type: "text",
                text: `${SEED_PROMPT_PREFIX} Here is the current graph...`,
              },
            ],
          },
          {
            id: "reply",
            role: "assistant",
            parts: [{ type: "text", text: "I see you have an empty graph." }],
          },
        ] as ReturnType<typeof useBuilderChatPanel>["messages"],
      }),
    );
    render(<BuilderChatPanel />);
    expect(screen.queryByText(SEED_PROMPT_PREFIX, { exact: false })).toBeNull();
    expect(screen.getByText("I see you have an empty graph.")).toBeDefined();
  });

  it("passes onGraphEdited and isGraphLoaded to useBuilderChatPanel", () => {
    const onGraphEdited = vi.fn();
    render(
      <BuilderChatPanel onGraphEdited={onGraphEdited} isGraphLoaded={true} />,
    );
    expect(mockUseBuilderChatPanel).toHaveBeenCalledWith(
      expect.objectContaining({ isGraphLoaded: true, onGraphEdited }),
    );
  });
});

describe("serializeGraphForChat", () => {
  it("returns empty message when no nodes", () => {
    const result = serializeGraphForChat([], []);
    expect(result).toBe("The graph is currently empty.");
  });

  it("lists block names and descriptions", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "Google Search",
          description: "Searches the web",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "block-1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
    ] as unknown as CustomNode[];

    const result = serializeGraphForChat(nodes, []);
    expect(result).toContain('"Google Search"');
    expect(result).toContain("Searches the web");
  });

  it("prefers metadata.customized_name over title", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "Original Title",
          description: "",
          metadata: { customized_name: "My Custom Name" },
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "block-1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
    ] as unknown as CustomNode[];

    const result = serializeGraphForChat(nodes, []);
    expect(result).toContain('"My Custom Name"');
    expect(result).not.toContain('"Original Title"');
  });

  it("truncates nodes beyond MAX_NODES limit", () => {
    const nodes = Array.from({ length: 110 }, (_, i) => ({
      id: String(i),
      data: {
        title: `Node ${i}`,
        description: "",
        hardcodedValues: {},
        inputSchema: {},
        outputSchema: {},
        uiType: 1,
        block_id: `block-${i}`,
        costs: [],
        categories: [],
      },
      type: "custom" as const,
      position: { x: 0, y: 0 },
    })) as unknown as CustomNode[];

    const result = serializeGraphForChat(nodes, []);
    expect(result).toContain("10 additional nodes not shown");
  });

  it("truncates edges beyond MAX_EDGES limit", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "A",
          description: "",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
      {
        id: "2",
        data: {
          title: "B",
          description: "",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b2",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 200, y: 0 },
      },
    ] as unknown as CustomNode[];

    const edges = Array.from({ length: 205 }, (_, i) => ({
      id: `e${i}`,
      source: "1",
      target: "2",
      sourceHandle: `out${i}`,
      targetHandle: `in${i}`,
      type: "custom" as const,
    })) as unknown as CustomEdge[];

    const result = serializeGraphForChat(nodes, edges);
    expect(result).toContain("5 additional connections not shown");
  });

  it("lists connections between nodes", () => {
    const nodes = [
      {
        id: "1",
        data: {
          title: "Search",
          description: "",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b1",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 0, y: 0 },
      },
      {
        id: "2",
        data: {
          title: "Formatter",
          description: "",
          hardcodedValues: {},
          inputSchema: {},
          outputSchema: {},
          uiType: 1,
          block_id: "b2",
          costs: [],
          categories: [],
        },
        type: "custom" as const,
        position: { x: 200, y: 0 },
      },
    ] as unknown as CustomNode[];

    const edges = [
      {
        id: "1:result->2:input",
        source: "1",
        target: "2",
        sourceHandle: "result",
        targetHandle: "input",
        type: "custom" as const,
      },
    ] as unknown as CustomEdge[];

    const result = serializeGraphForChat(nodes, edges);
    expect(result).toContain("Connections");
    expect(result).toContain('"Search"');
    expect(result).toContain('"Formatter"');
  });
});

describe("parseGraphActions", () => {
  it("returns empty array for plain text", () => {
    expect(parseGraphActions("This agent searches the web.")).toEqual([]);
  });

  it("parses update_node_input action", () => {
    const text = `
Here is a suggestion:
\`\`\`json
{"action": "update_node_input", "node_id": "1", "key": "query", "value": "AI news"}
\`\`\`
    `;
    const actions = parseGraphActions(text);
    expect(actions).toHaveLength(1);
    expect(actions[0]).toEqual({
      type: "update_node_input",
      nodeId: "1",
      key: "query",
      value: "AI news",
    });
  });

  it("parses connect_nodes action", () => {
    const text = `
\`\`\`json
{"action": "connect_nodes", "source": "1", "target": "2", "source_handle": "result", "target_handle": "input"}
\`\`\`
    `;
    const actions = parseGraphActions(text);
    expect(actions).toHaveLength(1);
    expect(actions[0]).toEqual({
      type: "connect_nodes",
      source: "1",
      target: "2",
      sourceHandle: "result",
      targetHandle: "input",
    });
  });

  it("parses multiple action blocks in a single message", () => {
    const text = `
Here are the changes:
\`\`\`json
{"action": "update_node_input", "node_id": "1", "key": "query", "value": "AI news"}
\`\`\`
\`\`\`json
{"action": "connect_nodes", "source": "1", "target": "2", "source_handle": "result", "target_handle": "input"}
\`\`\`
    `;
    const actions = parseGraphActions(text);
    expect(actions).toHaveLength(2);
    expect(actions[0].type).toBe("update_node_input");
    expect(actions[1].type).toBe("connect_nodes");
  });

  it("ignores invalid JSON blocks", () => {
    const text = "```json\nnot valid json\n```";
    expect(parseGraphActions(text)).toEqual([]);
  });

  it("ignores blocks without action field", () => {
    const text = '```json\n{"key": "value"}\n```';
    expect(parseGraphActions(text)).toEqual([]);
  });

  it("ignores update_node_input actions with missing required fields", () => {
    const text =
      '```json\n{"action": "update_node_input", "node_id": "1"}\n```';
    expect(parseGraphActions(text)).toEqual([]);
  });

  it("ignores connect_nodes actions with empty handles", () => {
    const text =
      '```json\n{"action": "connect_nodes", "source": "1", "target": "2", "source_handle": "", "target_handle": "input"}\n```';
    expect(parseGraphActions(text)).toEqual([]);
  });

  it("ignores update_node_input with non-primitive value", () => {
    const text =
      '```json\n{"action": "update_node_input", "node_id": "1", "key": "q", "value": {"nested": "object"}}\n```';
    expect(parseGraphActions(text)).toEqual([]);
  });

  it("accepts numeric and boolean primitive values", () => {
    const textNum =
      '```json\n{"action": "update_node_input", "node_id": "1", "key": "count", "value": 42}\n```';
    const textBool =
      '```json\n{"action": "update_node_input", "node_id": "1", "key": "enabled", "value": true}\n```';
    const numAction = parseGraphActions(textNum)[0];
    const boolAction = parseGraphActions(textBool)[0];
    expect(numAction?.type === "update_node_input" && numAction.value).toBe(42);
    expect(boolAction?.type === "update_node_input" && boolAction.value).toBe(
      true,
    );
  });
});

describe("getActionKey", () => {
  it("returns nodeId:key:value for update_node_input (includes value for multi-turn dedup)", () => {
    expect(
      getActionKey({
        type: "update_node_input",
        nodeId: "1",
        key: "query",
        value: "test",
      }),
    ).toBe('1:query:"test"');
  });

  it("generates distinct keys for same node+key but different values", () => {
    const key1 = getActionKey({
      type: "update_node_input",
      nodeId: "1",
      key: "query",
      value: "first",
    });
    const key2 = getActionKey({
      type: "update_node_input",
      nodeId: "1",
      key: "query",
      value: "corrected",
    });
    expect(key1).not.toBe(key2);
  });

  it("returns source:handle->target:handle for connect_nodes", () => {
    expect(
      getActionKey({
        type: "connect_nodes",
        source: "1",
        target: "2",
        sourceHandle: "result",
        targetHandle: "input",
      }),
    ).toBe("1:result->2:input");
  });
});

describe("getNodeDisplayName", () => {
  it("returns customized_name when set", () => {
    const node = {
      id: "1",
      data: {
        title: "Original",
        metadata: { customized_name: "My Custom" },
      },
    } as unknown as CustomNode;
    expect(getNodeDisplayName(node, "fallback")).toBe("My Custom");
  });

  it("falls back to title when no customized_name", () => {
    const node = {
      id: "1",
      data: { title: "Block Title" },
    } as unknown as CustomNode;
    expect(getNodeDisplayName(node, "fallback")).toBe("Block Title");
  });

  it("falls back to the provided fallback when node is undefined", () => {
    expect(getNodeDisplayName(undefined, "raw-id")).toBe("raw-id");
  });
});

describe("buildSeedPrompt", () => {
  it("starts with SEED_PROMPT_PREFIX", () => {
    const result = buildSeedPrompt("summary", "hello");
    expect(result.startsWith("I'm building an agent")).toBe(true);
  });

  it("wraps summary in <graph_context> tags", () => {
    const result = buildSeedPrompt("some graph summary", "hello");
    expect(result).toContain(
      "<graph_context>\nsome graph summary\n</graph_context>",
    );
  });

  it("includes format instructions for update_node_input", () => {
    const result = buildSeedPrompt("", "hello");
    expect(result).toContain('"action": "update_node_input"');
  });

  it("includes format instructions for connect_nodes", () => {
    const result = buildSeedPrompt("", "hello");
    expect(result).toContain('"action": "connect_nodes"');
  });

  it("ends with the user message appended", () => {
    const result = buildSeedPrompt("", "help me add a search block");
    expect(result).toContain("User request: help me add a search block");
  });
});

describe("extractTextFromParts", () => {
  it("returns empty string for empty array", () => {
    expect(extractTextFromParts([])).toBe("");
  });

  it("concatenates text parts in order", () => {
    const parts = [
      { type: "text", text: "Hello, " },
      { type: "text", text: "world!" },
    ];
    expect(extractTextFromParts(parts)).toBe("Hello, world!");
  });

  it("ignores non-text parts", () => {
    const parts = [
      { type: "text", text: "visible" },
      { type: "tool-call", text: "ignored" },
      { type: "text", text: " text" },
    ];
    expect(extractTextFromParts(parts)).toBe("visible text");
  });

  it("returns empty string when all parts are non-text", () => {
    const parts = [{ type: "tool-result" }, { type: "image" }];
    expect(extractTextFromParts(parts)).toBe("");
  });

  it("handles parts without a text field", () => {
    const parts = [{ type: "text" }, { type: "text", text: "hello" }];
    expect(extractTextFromParts(parts)).toBe("hello");
  });

  it("returns empty string for null parts", () => {
    expect(extractTextFromParts(null)).toBe("");
  });

  it("returns empty string for undefined parts", () => {
    expect(extractTextFromParts(undefined)).toBe("");
  });
});
