import { postV2CreateSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { getWebSocketToken } from "@/lib/supabase/actions";
import { environment } from "@/services/environment";
import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import { useEffect, useMemo, useRef, useState } from "react";
import { useEdgeStore } from "../../stores/edgeStore";
import { useNodeStore } from "../../stores/nodeStore";
import {
  buildSeedPrompt,
  extractTextFromParts,
  serializeGraphForChat,
} from "./helpers";

/**
 * Per-graph session cache.
 * Maps flowID → sessionId so the same chat session is reused each time the
 * user opens the panel for a given graph, preserving conversation history.
 * Lives at module scope to survive panel close/re-open without server round-trips.
 */
export const graphSessionCache = new Map<string, string>();

/** Clears the session cache. Exported only for use in tests. */
export function clearGraphSessionCacheForTesting() {
  graphSessionCache.clear();
}

interface UseSessionManagerArgs {
  isOpen: boolean;
  flowID: string | null;
  currentFlowIDRef: React.RefObject<string | null>;
}

/**
 * Manages the chat session lifecycle and transport for the builder chat panel.
 *
 * Responsibilities:
 * - Creates or reuses a per-graph chat session keyed by flowID.
 * - Resets all session state when the flowID changes (graph navigation).
 * - Builds a DefaultChatTransport once per session with per-request auth.
 * - Injects graph context into the first user message via prepareSendMessagesRequest.
 */
export function useSessionManager({
  isOpen,
  flowID,
  currentFlowIDRef,
}: UseSessionManagerArgs) {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isCreatingSession, setIsCreatingSession] = useState(false);
  const [sessionError, setSessionError] = useState(false);

  // Ref-based guard so the session-creation effect doesn't re-run (and cancel
  // the in-flight request) when setIsCreatingSession triggers a re-render.
  const isCreatingSessionRef = useRef(false);
  // Guards against injecting graph context more than once per session.
  const hasSentSeedMessageRef = useRef(false);

  // When the user navigates to a different graph: restore the cached session for
  // that graph (preserving the backend session) and reset all per-session state.
  // Messages are always cleared on navigation — appliedActionKeys cannot be persisted
  // so restoring messages while resetting action state would show previously applied
  // actions as unapplied, allowing them to be re-applied and creating duplicate undo entries.
  useEffect(() => {
    const cachedSessionId = flowID
      ? (graphSessionCache.get(flowID) ?? null)
      : null;
    setSessionId(cachedSessionId);
    setSessionError(false);
    isCreatingSessionRef.current = false;
    hasSentSeedMessageRef.current = false;
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
              let messageText = extractTextFromParts(last.parts ?? []);
              // Inject graph context into the first user message so the model
              // knows the current graph state without requiring an auto-sent
              // seed turn. hasSentSeedMessageRef is a stable ref so reading
              // .current here always reflects the latest value even though this
              // callback is created inside a useMemo.
              if (last.role === "user" && !hasSentSeedMessageRef.current) {
                hasSentSeedMessageRef.current = true;
                const edges = useEdgeStore.getState().edges;
                const currentNodes = useNodeStore.getState().nodes;
                const summary = serializeGraphForChat(currentNodes, edges);
                messageText = buildSeedPrompt(summary, messageText);
              }
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

  function retrySession() {
    if (flowID) graphSessionCache.delete(flowID);
    setSessionId(null);
    setSessionError(false);
    isCreatingSessionRef.current = false;
    hasSentSeedMessageRef.current = false;
    setMessages([]);
  }

  return {
    sessionId,
    isCreatingSession,
    sessionError,
    messages,
    setMessages,
    sendMessage,
    stop,
    status,
    error,
    retrySession,
    hasSentSeedMessageRef,
    isCreatingSessionRef,
  };
}
