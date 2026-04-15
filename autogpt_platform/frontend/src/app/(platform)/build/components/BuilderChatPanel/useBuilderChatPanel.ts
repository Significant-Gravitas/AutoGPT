import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { MarkerType } from "@xyflow/react";
import {
  type KeyboardEvent,
  type RefObject,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { parseAsString, useQueryStates } from "nuqs";
import { useShallow } from "zustand/react/shallow";
import { useEdgeStore } from "../../stores/edgeStore";
import { useNodeStore } from "../../stores/nodeStore";
import {
  GraphAction,
  buildSeedPrompt,
  extractTextFromParts,
  getActionKey,
  getNodeDisplayName,
  parseGraphActions,
  serializeGraphForChat,
} from "./helpers";

type SendMessageFn = ReturnType<typeof useChat>["sendMessage"];

/** Maximum number of undo entries to keep. Oldest entries are dropped when the limit is reached. */
const MAX_UNDO = 20;

/** Snapshot of node data taken before an action is applied, enabling undo. */
interface UndoSnapshot {
  actionKey: string;
  restore: () => void;
}

/**
 * Per-graph session cache.
 * Maps flowID → sessionId so the same chat session is reused each time the
 * user opens the panel for a given graph, preserving conversation history.
 * Lives at module scope to survive panel close/re-open without server round-trips.
 */
const graphSessionCache = new Map<string, string>();

/** Stable empty array so the useShallow selector returns the same reference when the panel is closed. */
const EMPTY_NODES: never[] = [];

/** Clears the session cache. Exported only for use in tests. */
export function clearGraphSessionCacheForTesting() {
  graphSessionCache.clear();
}

interface UseBuilderChatPanelArgs {
  isGraphLoaded?: boolean;
  onGraphEdited?: () => void;
  panelRef?: RefObject<HTMLElement | null>;
}

/**
 * Manages the lifecycle and state for the builder chat panel.
 *
 * Responsibilities:
 * - Session management: creates or reuses a per-graph chat session, keyed by
 *   flowID so reopening the panel for the same graph continues the conversation.
 * - Transport: builds a `DefaultChatTransport` once per session, with per-request
 *   auth token refresh via `getWebSocketToken`.
 * - Action parsing: extracts `update_node_input` and `connect_nodes` actions from
 *   completed assistant messages (gated on `status === "ready"`).
 * - Action application: applies validated graph mutations to Zustand stores,
 *   bypassing the global history to keep chat changes separate from Ctrl+Z.
 * - Tool detection: watches for completed `edit_agent` and `run_agent` tool calls
 *   to trigger graph reload and run auto-follow respectively.
 * - Undo: maintains a bounded LIFO stack (MAX_UNDO = 20) of restore callbacks.
 * - Input: owns the textarea value and keyboard shortcuts (Enter / Shift+Enter / Escape).
 */
export function useBuilderChatPanel({
  isGraphLoaded = false,
  onGraphEdited,
  panelRef,
}: UseBuilderChatPanelArgs = {}) {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [sessionError, setSessionError] = useState(false);
  const [appliedActionKeys, setAppliedActionKeys] = useState<Set<string>>(
    new Set(),
  );
  const [undoStack, setUndoStack] = useState<UndoSnapshot[]>([]);
  // Input state owned here to keep render logic out of the component.
  const [inputValue, setInputValue] = useState("");

  const sendMessageRef = useRef<SendMessageFn | null>(null);
  // Ref-based guard so the session-creation effect doesn't re-run (and cancel
  // the in-flight request) when setIsCreatingSession triggers a re-render.
  const isCreatingSessionRef = useRef(false);
  // Tracks tool call IDs already handled to avoid firing callbacks twice when
  // the messages array updates while status is "ready".
  const processedToolCallsRef = useRef(new Set<string>());
  // Guards against sending the seed message more than once per session.
  const hasSentSeedMessageRef = useRef(false);
  // Tracks the current flowID as a ref so in-flight session creation callbacks
  // can verify the graph hasn't changed before committing the new sessionId.
  const currentFlowIDRef = useRef<string | null>(null);

  const [{ flowID }, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowExecutionID: parseAsString,
  });
  // Keep ref in sync with the current flowID so in-flight session callbacks can
  // detect stale graph context without closure staleness issues.
  currentFlowIDRef.current = flowID;
  const { toast } = useToast();

  const nodes = useNodeStore(
    useShallow((s) => (isOpen ? s.nodes : EMPTY_NODES)),
  );
  const setNodes = useNodeStore((s) => s.setNodes);
  const setEdges = useEdgeStore((s) => s.setEdges);

  // When the user navigates to a different graph: restore the cached session for
  // that graph (preserving the backend session) and reset all per-session UI state.
  // Messages are always cleared on navigation — appliedActionKeys cannot be persisted
  // so restoring messages while resetting action state would show previously applied
  // actions as unapplied, allowing them to be re-applied and creating duplicate undo entries.
  useEffect(() => {
    const cachedSessionId = flowID
      ? (graphSessionCache.get(flowID) ?? null)
      : null;
    setSessionId(cachedSessionId);
    setSessionError(false);
    setAppliedActionKeys(new Set());
    setUndoStack([]);
    setInputValue("");
    isCreatingSessionRef.current = false;
    processedToolCallsRef.current = new Set();
    hasSentSeedMessageRef.current = false;
    setMessages([]);
    // setMessages is a stable function from useChat; excluding from deps is safe.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flowID]);

  // Create a new chat session when the panel opens and no session exists yet.
  useEffect(() => {
    if (!isOpen || sessionId || isCreatingSessionRef.current || sessionError)
      return;
    // The `cancelled` flag prevents state updates after the component unmounts
    // or the effect re-runs, avoiding stale state from async calls.
    let cancelled = false;
    isCreatingSessionRef.current = true;
    // Snapshot the flowID at effect start so the result is rejected if the
    // user navigates to a different graph before the request completes, preventing
    // the old session from being assigned to the new graph.
    const effectFlowID = flowID;

    async function createSession() {
      setIsCreatingSession(true);
      try {
        // NOTE: The backend validates that the authenticated user owns the
        // session before allowing any messages — session IDs alone are not
        // sufficient for unauthorized access.
        const res = await postV2CreateSession(null);
        // Discard the result if the effect was cancelled (unmount or re-run) or
        // if the user navigated to a different graph before the request completed.
        if (cancelled || currentFlowIDRef.current !== effectFlowID) return;
        if (res.status === 200) {
          const id = res.data.id;
          // Validate the session ID is a safe non-empty identifier before
          // interpolating it into the streaming URL — rejects values that
          // contain path-traversal characters or whitespace.
          if (typeof id !== "string" || !id || !/^[\w-]+$/i.test(id)) {
            setSessionError(true);
            return;
          }
          setSessionId(id);
          // Cache so this session is reused next time the same graph is opened.
          if (effectFlowID) graphSessionCache.set(effectFlowID, id);
        } else {
          setSessionError(true);
        }
      } catch {
        if (!cancelled) setSessionError(true);
      } finally {
        if (!cancelled) {
          setIsCreatingSession(false);
          isCreatingSessionRef.current = false;
        }
      }
    }

    createSession();
    return () => {
      cancelled = true;
      isCreatingSessionRef.current = false;
    };
    // isCreatingSession is intentionally excluded: the ref guards re-entry so
    // state-driven re-renders don't cancel the in-flight request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, sessionId, sessionError]);

  const transport = useMemo(
    () =>
      sessionId
        ? new DefaultChatTransport({
            api: `${environment.getAGPTServerBaseUrl()}/api/chat/sessions/${sessionId}/stream`,
            prepareSendMessagesRequest: async ({ messages }) => {
              const last = messages.at(-1);
              if (!last)
                throw new Error(
                  "No message to send — messages array is empty.",
                );
              const { token, error } = await getWebSocketToken();
              if (error || !token)
                throw new Error(
                  "Authentication failed — please sign in again.",
                );
              const messageText = extractTextFromParts(last.parts ?? []);
              return {
                body: {
                  message: messageText,
                  is_user_message: last.role === "user",
                  context: null,
                  file_ids: null,
                  mode: null,
                },
                headers: { Authorization: `Bearer ${token}` },
              };
            },
          })
        : null,
    [sessionId],
  );

  const { messages, setMessages, sendMessage, stop, status, error } = useChat({
    id: sessionId ?? undefined,
    transport: transport ?? undefined,
  });

  // Keep a stable ref so callbacks can call sendMessage without it appearing
  // in their dependency arrays.
  sendMessageRef.current = sendMessage;

  // Send the seed message once per session when the session becomes available
  // and the graph is loaded. The ref guard prevents duplicate sends when the
  // effect re-runs due to dependency changes.
  useEffect(() => {
    if (!sessionId || !isGraphLoaded || hasSentSeedMessageRef.current) return;
    hasSentSeedMessageRef.current = true;
    const edges = useEdgeStore.getState().edges;
    const summary = serializeGraphForChat(nodes, edges);
    sendMessageRef.current?.({ text: buildSeedPrompt(summary) });
    // nodes is intentionally excluded: the seed only fires once per session and
    // reading the live value here is sufficient. edges are read via getState().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, isGraphLoaded]);

  // Parsed actions from all assistant messages, accumulated across turns.
  // Gated on `status === "ready"` so parsing only runs on completed turns.
  const parsedActions = useMemo(() => {
    if (status !== "ready") return [];
    const seen = new Set<string>();
    return messages
      .filter((m) => m.role === "assistant")
      .flatMap((msg) => parseGraphActions(extractTextFromParts(msg.parts)))
      .filter((action) => {
        const key = getActionKey(action);
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      });
  }, [messages, status]);

  // Detect completed edit_agent and run_agent tool calls and act on them.
  // edit_agent → trigger a graph reload via the onGraphEdited callback.
  // run_agent  → update flowExecutionID in the URL to auto-follow the new run.
  useEffect(() => {
    if (status !== "ready") return;
    for (const msg of messages) {
      if (msg.role !== "assistant") continue;
      for (const part of msg.parts ?? []) {
        if (part.type !== "dynamic-tool") continue;
        const dynPart = part as {
          type: "dynamic-tool";
          toolName: string;
          toolCallId: string;
          state: string;
          output?: unknown;
        };
        if (dynPart.state !== "output-available") continue;
        if (processedToolCallsRef.current.has(dynPart.toolCallId)) continue;
        processedToolCallsRef.current.add(dynPart.toolCallId);

        if (dynPart.toolName === "edit_agent") {
          onGraphEdited?.();
        } else if (dynPart.toolName === "run_agent") {
          const output = dynPart.output as Record<string, unknown> | null;
          const execId = output?.execution_id;
          if (typeof execId === "string" && /^[\w-]+$/i.test(execId)) {
            setQueryStates({ flowExecutionID: execId });
          }
        }
      }
    }
  }, [messages, status, onGraphEdited, setQueryStates]);

  // Close the panel on Escape when focus is inside the panel, so pressing Escape
  // in another dialog or canvas element does not accidentally close the chat panel.
  // Skip when focus is in an editable element to avoid discarding a draft in progress.
  useEffect(() => {
    if (!isOpen) return;
    function onKeyDown(e: globalThis.KeyboardEvent) {
      if (e.key !== "Escape") return;
      if (
        panelRef &&
        panelRef.current &&
        !panelRef.current.contains(e.target as Node)
      )
        return;
      const target = e.target as HTMLElement;
      if (
        target.tagName === "TEXTAREA" ||
        target.tagName === "INPUT" ||
        target.isContentEditable
      )
        return;
      setIsOpen(false);
    }
    document.addEventListener("keydown", onKeyDown);
    return () => document.removeEventListener("keydown", onKeyDown);
  }, [isOpen, panelRef]);

  const isStreaming = status === "streaming" || status === "submitted";
  const canSend =
    Boolean(sessionId) && !isCreatingSession && !sessionError && !isStreaming;

  function handleToggle() {
    setIsOpen((o) => !o);
  }

  // Resets session error state so the session-creation effect re-runs on
  // the next render without toggling the panel closed and back open.
  // Also evicts the stale cached session so a fresh one is created.
  // hasSentSeedMessageRef is reset so the seed message is re-sent to the
  // new session (it may have been set to true by a previous successful session
  // that was later invalidated without a flowID change).
  // Messages are cleared so stale messages from the previous session are not
  // shown alongside content from the new session.
  function retrySession() {
    if (flowID) graphSessionCache.delete(flowID);
    setSessionId(null);
    setSessionError(false);
    isCreatingSessionRef.current = false;
    hasSentSeedMessageRef.current = false;
    setMessages([]);
  }

  function handleSend() {
    const text = inputValue.trim();
    if (!text || !canSend) return;
    setInputValue("");
    sendMessage({ text });
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function handleApplyAction(action: GraphAction) {
    if (action.type === "update_node_input") {
      // Read live state for both validation and mutation so rapid successive
      // applies see the latest nodes rather than a stale render-cycle snapshot.
      const liveNodes = useNodeStore.getState().nodes;
      const node = liveNodes.find((n) => n.id === action.nodeId);
      if (!node) {
        toast({
          title: "Cannot apply change",
          description: `Node "${action.nodeId}" was not found in the graph.`,
          variant: "destructive",
        });
        return;
      }
      // Block prototype-polluting keys regardless of schema presence.
      // The schema check below uses hasOwnProperty so __proto__ is caught when
      // schemaProps exists, but this guard handles the no-schema case.
      const DANGEROUS_KEYS = ["__proto__", "constructor", "prototype"];
      if (DANGEROUS_KEYS.includes(action.key)) {
        toast({
          title: "Cannot apply change",
          description: `Field "${action.key}" is not a valid input.`,
          variant: "destructive",
        });
        return;
      }
      // Reject keys not present in the node's input schema to prevent writing
      // arbitrary fields that the block does not support.
      const schemaProps = node.data.inputSchema?.properties;
      if (
        schemaProps &&
        !Object.prototype.hasOwnProperty.call(schemaProps, action.key)
      ) {
        toast({
          title: "Cannot apply change",
          description: `Field "${action.key}" is not a valid input for "${getNodeDisplayName(node, node.id)}".`,
          variant: "destructive",
        });
        return;
      }
      // Capture a shallow-copied nodes snapshot before mutating. Spreading
      // ensures the undo restore references an independent array rather than
      // the same reference that the store may update in-place.
      // Both the apply and the restore use setNodes (not updateNodeData) to
      // bypass the global history store — this keeps chat-panel changes
      // completely separate from Ctrl+Z, preventing the "Applied" badge from
      // going stale after a global undo.
      const prevNodes = [...liveNodes];
      const nextNodes = liveNodes.map((n) =>
        n.id === action.nodeId
          ? {
              ...n,
              data: {
                ...n.data,
                hardcodedValues: {
                  ...n.data.hardcodedValues,
                  [action.key]: action.value,
                },
              },
            }
          : n,
      );
      const key = getActionKey(action);
      setUndoStack((prev) => {
        const entry: UndoSnapshot = {
          actionKey: key,
          restore: () => {
            setNodes(prevNodes);
            setAppliedActionKeys((keys) => {
              const next = new Set(keys);
              next.delete(key);
              return next;
            });
          },
        };
        const trimmed = prev.length >= MAX_UNDO ? prev.slice(1) : prev;
        return [...trimmed, entry];
      });
      setNodes(nextNodes);
    } else if (action.type === "connect_nodes") {
      // Read live state so validation reflects the current graph even when
      // multiple actions are applied within the same render cycle.
      const liveNodes = useNodeStore.getState().nodes;
      const sourceNode = liveNodes.find((n) => n.id === action.source);
      const targetNode = liveNodes.find((n) => n.id === action.target);
      if (!sourceNode || !targetNode) {
        toast({
          title: "Cannot apply connection",
          description: `One or both nodes (${action.source}, ${action.target}) were not found.`,
          variant: "destructive",
        });
        return;
      }
      // Validate that the referenced handles exist on the respective nodes.
      const srcProps = sourceNode.data.outputSchema?.properties;
      const tgtProps = targetNode.data.inputSchema?.properties;
      if (
        srcProps &&
        !Object.prototype.hasOwnProperty.call(srcProps, action.sourceHandle)
      ) {
        toast({
          title: "Cannot apply connection",
          description: `Output handle "${action.sourceHandle}" does not exist on "${getNodeDisplayName(sourceNode, action.source)}".`,
          variant: "destructive",
        });
        return;
      }
      if (
        tgtProps &&
        !Object.prototype.hasOwnProperty.call(tgtProps, action.targetHandle)
      ) {
        toast({
          title: "Cannot apply connection",
          description: `Input handle "${action.targetHandle}" does not exist on "${getNodeDisplayName(targetNode, action.target)}".`,
          variant: "destructive",
        });
        return;
      }
      const edgeId = `${action.source}:${action.sourceHandle}->${action.target}:${action.targetHandle}`;
      // Shallow-copy the edges snapshot so the undo restore references an
      // independent array rather than the same reference the store may update.
      // Both the apply and the restore use setEdges (not addEdge/removeEdge)
      // to bypass the global history store — keeps chat-panel changes separate.
      const prevEdges = [...useEdgeStore.getState().edges];
      // Guard against duplicate edges — the same connection may appear after an
      // undo-then-reapply or from identical suggestions across AI messages.
      const alreadyExists = prevEdges.some(
        (e) =>
          e.source === action.source &&
          e.target === action.target &&
          e.sourceHandle === action.sourceHandle &&
          e.targetHandle === action.targetHandle,
      );
      if (alreadyExists) {
        // Edge already present — mark as applied without duplicating it.
        setAppliedActionKeys((prev) => {
          const next = new Set(prev);
          next.add(getActionKey(action));
          return next;
        });
        return;
      }
      const key = getActionKey(action);
      setUndoStack((prev) => {
        const entry: UndoSnapshot = {
          actionKey: key,
          restore: () => {
            setEdges(prevEdges);
            setAppliedActionKeys((keys) => {
              const next = new Set(keys);
              next.delete(key);
              return next;
            });
          },
        };
        const trimmed = prev.length >= MAX_UNDO ? prev.slice(1) : prev;
        return [...trimmed, entry];
      });
      setEdges([
        ...prevEdges,
        {
          id: edgeId,
          source: action.source,
          target: action.target,
          sourceHandle: action.sourceHandle,
          targetHandle: action.targetHandle,
          type: "custom",
          // Match the markerEnd style used by addEdge in edgeStore so
          // chat-applied edges render with the same arrowhead as manually drawn ones.
          markerEnd: {
            type: MarkerType.ArrowClosed,
            strokeWidth: 2,
            color: "#555",
          },
        },
      ]);
    } else {
      // Exhaustiveness guard — TypeScript ensures all GraphAction types are handled above.
      const _: never = action;
      return _;
    }
    setAppliedActionKeys((prev) => {
      const next = new Set(prev);
      next.add(getActionKey(action));
      return next;
    });
  }

  function handleUndoLastAction() {
    // Read the current stack directly rather than inside the setUndoStack updater.
    // Calling restore() (which triggers setNodes/setEdges) inside a state updater
    // is a React anti-pattern — state updaters must be pure. Reading from the ref
    // here is safe because this function is only called from event handlers.
    const stack = undoStack;
    if (stack.length === 0) return;
    const last = stack[stack.length - 1];
    last.restore();
    setUndoStack((prev) => prev.slice(0, -1));
  }

  // Sends an arbitrary text message directly, bypassing the input field.
  // Used by CopilotChatActionsProvider so tool components (e.g. EditAgentTool)
  // can programmatically send "try again" prompts without touching the textarea.
  function sendRawMessage(text: string) {
    if (!text || !canSend) return;
    sendMessage({ text });
  }

  return {
    isOpen,
    handleToggle,
    retrySession,
    messages,
    stop,
    error,
    isCreatingSession,
    sessionError,
    sessionId,
    nodes,
    parsedActions,
    appliedActionKeys,
    handleApplyAction,
    undoStack,
    handleUndoLastAction,
    // Input handling (owned here to keep component render-only)
    inputValue,
    setInputValue,
    handleSend,
    sendRawMessage,
    handleKeyDown,
    isStreaming,
    canSend,
  };
}
