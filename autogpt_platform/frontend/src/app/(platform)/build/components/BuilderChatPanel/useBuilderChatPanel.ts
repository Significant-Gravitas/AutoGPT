import { useToast } from "@/components/molecules/Toast/use-toast";
import {
  type KeyboardEvent,
  type RefObject,
  useEffect,
  useRef,
  useState,
} from "react";
import { parseAsString, useQueryStates } from "nuqs";
import { useShallow } from "zustand/react/shallow";
import { useEdgeStore } from "../../stores/edgeStore";
import { useNodeStore } from "../../stores/nodeStore";
import {
  applyConnectNodes,
  applyUpdateNodeInput,
  type UndoSnapshot,
} from "./actionApplicators";
import { GraphAction, getActionKey } from "./helpers";
import { useActionParser } from "./useActionParser";
import {
  clearGraphSessionCacheForTesting,
  graphSessionCache,
  useSessionManager,
} from "./useSessionManager";
import { useToolCallHandler } from "./useToolCallHandler";

export { clearGraphSessionCacheForTesting };

/** Stable empty array so the useShallow selector returns the same reference when the panel is closed. */
const EMPTY_NODES: never[] = [];

interface UseBuilderChatPanelArgs {
  isGraphLoaded?: boolean;
  onGraphEdited?: () => void;
  panelRef?: RefObject<HTMLElement | null>;
}

/**
 * Thin coordinator for the builder chat panel.
 *
 * Delegates to focused sub-hooks:
 * - `useSessionManager` — session lifecycle, transport, and useChat integration.
 * - `useActionParser`   — action parsing from completed assistant messages.
 * - `useToolCallHandler` — edit_agent / run_agent tool call detection.
 *
 * Owns the remaining concerns directly: panel open/close state, action
 * application with undo, and input handling.
 */
export function useBuilderChatPanel({
  onGraphEdited,
  panelRef,
}: UseBuilderChatPanelArgs = {}) {
  const [isOpen, setIsOpen] = useState(false);
  const [appliedActionKeys, setAppliedActionKeys] = useState<Set<string>>(
    new Set(),
  );
  const [undoStack, setUndoStack] = useState<UndoSnapshot[]>([]);
  const [inputValue, setInputValue] = useState("");

  // Tracks the current flowID as a ref so in-flight session creation callbacks
  // can verify the graph hasn't changed before committing the new sessionId.
  const currentFlowIDRef = useRef<string | null>(null);

  const [{ flowID }] = useQueryStates({
    flowID: parseAsString,
    flowExecutionID: parseAsString,
  });
  // Keep ref in sync with the current flowID so in-flight session callbacks can
  // detect stale graph context without closure staleness issues.
  // Assigned during render (not in useEffect) so callbacks created in the same
  // render cycle see the current flowID immediately — useEffect would delay by
  // one render, creating a window where stale values are visible.
  currentFlowIDRef.current = flowID;
  const { toast } = useToast();

  const nodes = useNodeStore(
    useShallow((s) => (isOpen ? s.nodes : EMPTY_NODES)),
  );
  const setNodes = useNodeStore((s) => s.setNodes);
  const setEdges = useEdgeStore((s) => s.setEdges);

  const {
    sessionId,
    isCreatingSession,
    sessionError,
    messages,
    setMessages,
    sendMessage,
    stop,
    status,
    error,
    retrySession: retrySessionBase,
    isCreatingSessionRef,
  } = useSessionManager({ isOpen, flowID, currentFlowIDRef });

  const { parsedActions } = useActionParser({ messages, status });

  useToolCallHandler({ messages, status, flowID, onGraphEdited });

  // When the user navigates to a different graph: reset all per-session UI state.
  // Messages are always cleared on navigation — appliedActionKeys cannot be persisted
  // so restoring messages while resetting action state would show previously applied
  // actions as unapplied, allowing them to be re-applied and creating duplicate undo entries.
  useEffect(() => {
    setAppliedActionKeys(new Set());
    setUndoStack([]);
    setInputValue("");
    isCreatingSessionRef.current = false;
    setMessages([]);
    // setMessages is a stable function from useChat; excluding from deps is safe.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [flowID]);

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
  // Messages are cleared so stale messages from the previous session are not
  // shown alongside content from the new session.
  function retrySession() {
    if (flowID) graphSessionCache.delete(flowID);
    retrySessionBase();
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
    const deps = {
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
      return _;
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
  // Silently truncates to the backend's 64,000-character limit to avoid a 422.
  function sendRawMessage(text: string) {
    if (!text || !canSend) return;
    const MAX_RAW_MESSAGE_CHARS = 64_000;
    const safeText =
      text.length > MAX_RAW_MESSAGE_CHARS
        ? text.slice(0, MAX_RAW_MESSAGE_CHARS)
        : text;
    sendMessage({ text: safeText });
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
