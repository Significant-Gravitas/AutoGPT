import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
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
  ApplyActionDeps,
  UndoSnapshot,
  applyConnectNodes,
  applyUpdateNodeInput,
} from "./actionApplicators";
import {
  GraphAction,
  buildSeedPrompt,
  extractTextFromParts,
  getActionKey,
  parseGraphActions,
  serializeGraphForChat,
} from "./helpers";
import { TEXTAREA_MAX_LENGTH } from "./components/PanelInput";

type SendMessageFn = ReturnType<typeof useChat>["sendMessage"];

/**
 * Per-graph session cache with a simple LRU cap.
 * Maps flowID → sessionId so the same chat session is reused each time the
 * user opens the panel for a given graph, preserving conversation history.
 * Lives at module scope to survive panel close/re-open without server round-trips.
 *
 * JavaScript `Map` preserves insertion order, so we implement LRU by deleting
 * and re-inserting on access, and evicting the oldest entry when over `MAX_SESSION_CACHE`.
 */
const MAX_SESSION_CACHE = 50;
const graphSessionCache = new Map<string, string>();

function cacheGetSession(flowID: string): string | undefined {
  const id = graphSessionCache.get(flowID);
  if (id !== undefined) {
    // Move to most-recent position.
    graphSessionCache.delete(flowID);
    graphSessionCache.set(flowID, id);
  }
  return id;
}

function cacheSetSession(flowID: string, sessionId: string): void {
  if (graphSessionCache.has(flowID)) {
    graphSessionCache.delete(flowID);
  } else if (graphSessionCache.size >= MAX_SESSION_CACHE) {
    const oldestKey = graphSessionCache.keys().next().value;
    if (oldestKey !== undefined) graphSessionCache.delete(oldestKey);
  }
  graphSessionCache.set(flowID, sessionId);
}

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
 *   completed assistant messages (gated on `status === "ready"`). Parsing is
 *   incremental — only newly added messages are re-scanned each turn.
 * - Action application: delegates to helpers in `actionApplicators.ts` that
 *   validate and apply graph mutations to Zustand stores, bypassing the global
 *   history to keep chat changes separate from Ctrl+Z.
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
  // Parsed actions accumulated across completed assistant turns. Kept in state
  // (rather than derived via useMemo) because the incremental cache mutates
  // refs — mutating inside a memo would break React Strict Mode, which runs
  // memos twice and would cause the second pass to skip messages.
  const [parsedActions, setParsedActions] = useState<GraphAction[]>([]);
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
  // Tracks the highest message index already scanned for actions so subsequent
  // turns only re-parse new assistant messages instead of O(all_messages).
  const lastParsedMessageIndexRef = useRef(-1);
  // Mirrors lastParsedMessageIndexRef but for tool-call detection, so the
  // tool-call effect is also O(new_messages) not O(all_messages).
  const lastScannedToolCallIndexRef = useRef(-1);
  // Cached deduplicated action list that survives across re-renders so that
  // incremental parsing can merge new actions into it without a full re-scan.
  const parsedActionsCacheRef = useRef<{
    actions: GraphAction[];
    seen: Set<string>;
  }>({ actions: [], seen: new Set() });
  // Navigation race guard: set by the flowID-reset effect when an *actual*
  // graph navigation occurs (not initial mount). The parse-actions effect
  // checks this flag and skips one pass, because the cleanup effect's
  // `setMessages([])` is queued and not yet committed when parse-actions
  // runs in the same effect cycle — without the skip, we'd re-scan the
  // previous graph's messages from index 0 (refs were just reset) and
  // briefly flash old action buttons in the new graph's panel.
  const skipNextParseRef = useRef(false);
  // Tracks the previous flowID so the reset effect can distinguish initial
  // mount (no skip needed — fresh hook, no stale messages) from real
  // navigation (skip needed — closure has prior-graph messages).
  const prevFlowIDRef = useRef<string | null>(null);

  const [{ flowID }, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowExecutionID: parseAsString,
  });
  // Keep ref in sync with the current flowID so in-flight session callbacks can
  // detect stale graph context without closure staleness issues. Using an
  // effect rather than a render-body write keeps the render pure.
  useEffect(() => {
    currentFlowIDRef.current = flowID;
  }, [flowID]);
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
    // Detect actual navigation (not initial mount) so the parse-actions
    // effect can skip its next pass — see ``skipNextParseRef`` declaration
    // for the race-condition rationale.
    const isNavigation =
      prevFlowIDRef.current !== null && prevFlowIDRef.current !== flowID;
    prevFlowIDRef.current = flowID;
    if (isNavigation) {
      skipNextParseRef.current = true;
    }

    const cachedSessionId = flowID ? (cacheGetSession(flowID) ?? null) : null;
    setSessionId(cachedSessionId);
    setSessionError(false);
    setAppliedActionKeys(new Set());
    setUndoStack([]);
    setInputValue("");
    setParsedActions([]);
    isCreatingSessionRef.current = false;
    processedToolCallsRef.current = new Set();
    hasSentSeedMessageRef.current = false;
    lastParsedMessageIndexRef.current = -1;
    lastScannedToolCallIndexRef.current = -1;
    parsedActionsCacheRef.current = { actions: [], seen: new Set() };
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
          if (effectFlowID) cacheSetSession(effectFlowID, id);
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

  // Memoised so the same DefaultChatTransport instance is reused across
  // re-renders (e.g. every streaming chunk triggers a render). Recreating it
  // on each render resets useChat's internal Chat instance mid-stream, causing
  // the streaming connection to break. Only recreate when sessionId changes.
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

  // Send the seed message once per session when the panel is open, the session
  // becomes available, and the graph is loaded. The ref guard prevents duplicate
  // sends when the effect re-runs due to dependency changes.
  // isOpen is required: when the panel is closed, the nodes selector returns EMPTY_NODES
  // to avoid unnecessary store subscriptions. Sending a seed with an empty graph would
  // poison the AI context, so we defer until the panel is actually visible.
  useEffect(() => {
    if (
      !isOpen ||
      !sessionId ||
      !isGraphLoaded ||
      hasSentSeedMessageRef.current
    )
      return;
    hasSentSeedMessageRef.current = true;
    const edges = useEdgeStore.getState().edges;
    const summary = serializeGraphForChat(nodes, edges);
    sendMessageRef.current?.({ text: buildSeedPrompt(summary) });
    // nodes is intentionally excluded: the seed only fires once per session and
    // reading the live value here is sufficient. edges are read via getState().
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, sessionId, isGraphLoaded]);

  // Parsed actions from all assistant messages, accumulated across turns.
  // Gated on `status === "ready"` so parsing only runs on completed turns.
  // Uses an incremental cache keyed off `lastParsedMessageIndexRef` so each
  // completed turn only re-scans the newly added messages rather than the
  // entire conversation history.
  //
  // This lives in an effect (not a memo) because ref mutation inside a memo
  // body would break React Strict Mode: memos can run twice and the second
  // pass would see `lastParsedMessageIndexRef` already advanced and skip the
  // new messages, losing actions.
  useEffect(() => {
    if (status !== "ready") return;
    // Navigation race guard: the flowID-reset effect above signals a graph
    // navigation by setting `skipNextParseRef`. The cleanup runs first in the
    // effect cycle and resets `lastParsedMessageIndexRef` + the cache, but the
    // `setMessages([])` it queues is not committed until the next render —
    // so this effect's `messages` closure still belongs to the previous graph.
    // Skipping one pass prevents a re-scan of stale messages from index 0
    // (which would flash the prior graph's actions in the new panel).
    if (skipNextParseRef.current) {
      skipNextParseRef.current = false;
      return;
    }
    const cache = parsedActionsCacheRef.current;
    const startIndex = lastParsedMessageIndexRef.current + 1;
    let appendedAny = false;
    for (let i = startIndex; i < messages.length; i++) {
      const msg = messages[i];
      if (msg.role !== "assistant") continue;
      const newActions = parseGraphActions(extractTextFromParts(msg.parts));
      for (const action of newActions) {
        const key = getActionKey(action);
        if (cache.seen.has(key)) continue;
        cache.seen.add(key);
        cache.actions.push(action);
        appendedAny = true;
      }
    }
    lastParsedMessageIndexRef.current = messages.length - 1;
    if (appendedAny) {
      // Fresh array reference so consumers re-render with the new actions.
      setParsedActions([...cache.actions]);
    }
  }, [messages, status, flowID]);

  // Detect completed edit_agent and run_agent tool calls and act on them.
  // edit_agent → trigger a graph reload via the onGraphEdited callback.
  // run_agent  → update flowExecutionID in the URL to auto-follow the new run.
  // Uses lastScannedToolCallIndexRef to mirror the action parser's incremental
  // approach — only newly added messages are scanned each turn.
  useEffect(() => {
    if (status !== "ready") return;
    const startIndex = lastScannedToolCallIndexRef.current + 1;
    for (let i = startIndex; i < messages.length; i++) {
      const msg = messages[i];
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
    lastScannedToolCallIndexRef.current = messages.length - 1;
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
    lastParsedMessageIndexRef.current = -1;
    lastScannedToolCallIndexRef.current = -1;
    parsedActionsCacheRef.current = { actions: [], seen: new Set() };
    setParsedActions([]);
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
    const deps: ApplyActionDeps = {
      toast,
      setNodes,
      setEdges,
      setUndoStack,
      setAppliedActionKeys,
    };
    let applied = false;
    if (action.type === "update_node_input") {
      applied = applyUpdateNodeInput(action, deps);
    } else if (action.type === "connect_nodes") {
      applied = applyConnectNodes(action, deps);
    } else {
      // Exhaustiveness guard — TypeScript ensures all GraphAction types are handled above.
      const _: never = action;
      void _;
    }
    if (applied) {
      setAppliedActionKeys((prev) => {
        const next = new Set(prev);
        next.add(getActionKey(action));
        return next;
      });
    }
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
  // Enforces the same length cap as the visible textarea so programmatic callers
  // cannot bypass the limit.
  function sendRawMessage(text: string) {
    if (!text || !canSend) return;
    const trimmed =
      text.length > TEXTAREA_MAX_LENGTH
        ? text.slice(0, TEXTAREA_MAX_LENGTH)
        : text;
    sendMessage({ text: trimmed });
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
