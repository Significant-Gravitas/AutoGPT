import {
  getGetV1GetSpecificGraphQueryKey,
  useGetV1GetSpecificGraph,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  usePostV1CreateNewGraph,
  usePutV1SetActiveGraphVersion,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import {
  getGetV2GetSessionQueryKey,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { parseAsInteger, parseAsString, useQueryStates } from "nuqs";
import { useEffect, useMemo, useRef, useState } from "react";
import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { useCopilotStream } from "@/app/(platform)/copilot/useCopilotStream";
import { useCopilotPendingChips } from "@/app/(platform)/copilot/useCopilotPendingChips";
import { useGetV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";

interface UseBuilderChatPanelArgs {
  panelRef?: React.RefObject<HTMLElement | null>;
}

type UiMessages = UIMessage<unknown, UIDataTypes, UITools>[];

/**
 * Normalize a tool part's `output` to a plain object.
 *
 * On the live AI-SDK stream the backend encodes tool outputs as JSON strings
 * (see `stash_pending_tool_output` on the backend — dicts get `json.dumps`d
 * before being sent). On hydration from DB the session-converter already
 * parsed that string back to an object. So this effect may see either shape,
 * and we need a tolerant reader. Returns null if the value doesn't
 * resemble a structured response (e.g. still a primitive partial chunk).
 */
function parseToolOutput(raw: unknown): Record<string, unknown> | null {
  if (raw == null) return null;
  if (typeof raw === "object" && !Array.isArray(raw)) {
    return raw as Record<string, unknown>;
  }
  if (typeof raw === "string") {
    const trimmed = raw.trim();
    if (!trimmed.startsWith("{")) return null;
    try {
      const parsed = JSON.parse(trimmed) as unknown;
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        return parsed as Record<string, unknown>;
      }
    } catch {
      // Mid-stream partial JSON — swallow and wait for the completion event.
    }
  }
  return null;
}

export function useBuilderChatPanel({
  panelRef,
}: UseBuilderChatPanelArgs = {}) {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [revertTargetVersion, setRevertTargetVersion] = useState<number | null>(
    null,
  );
  // Retry tokens: bumping forces the bind / bootstrap effect to re-run after
  // a failure so the panel can recover without a close+reopen round-trip.
  const [bindRetryToken, setBindRetryToken] = useState(0);
  const [bootstrapRetryToken, setBootstrapRetryToken] = useState(0);
  // Non-null when the corresponding async op failed; drives the retry UI
  // surfaced by the panel.  Cleared on each retry attempt and on success.
  const [bindError, setBindError] = useState<string | null>(null);
  const [bootstrapError, setBootstrapError] = useState<string | null>(null);
  const [{ flowID, flowVersion }, setQueryStates] = useQueryStates({
    flowID: parseAsString,
    flowExecutionID: parseAsString,
    flowVersion: parseAsInteger,
  });
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: graph, refetch: refetchGraph } = useGetV1GetSpecificGraph(
    flowID ?? "",
    {},
    {
      query: {
        select: okData,
        enabled: !!flowID,
      },
    },
  );

  // Unified /sessions endpoint: setting ``builder_graph_id`` routes the
  // request through the get-or-create path keyed on (user_id, graph_id)
  // so the panel re-binds to the same session across refreshes.
  const { mutateAsync: createBuilderSession } = usePostV2CreateSession();
  const { mutateAsync: createNewGraph, isPending: isBootstrappingGraph } =
    usePostV1CreateNewGraph();
  const { mutateAsync: setActiveVersion } = usePutV1SetActiveGraphVersion();

  const sessionQuery = useGetV2GetSession(sessionId ?? "", undefined, {
    query: {
      enabled: !!sessionId,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnMount: true,
    },
  });

  const hasActiveStream =
    sessionQuery.data?.status === 200
      ? !!sessionQuery.data.data.active_stream
      : false;

  // Memoize so the hydration effect in useCopilotStream doesn't infinite-loop
  // on a new array reference every render. Re-derives only when query data,
  // session id, or stream-active state changes.
  const hydratedMessages = useMemo<UiMessages | undefined>(() => {
    if (sessionQuery.data?.status !== 200 || !sessionId) return undefined;
    return convertChatSessionMessagesToUiMessages(
      sessionId,
      sessionQuery.data.data.messages ?? [],
      { isComplete: !hasActiveStream },
    ).messages as UiMessages;
  }, [sessionQuery.data, sessionId, hasActiveStream]);

  const { messages, setMessages, sendMessage, stop, status, error } =
    useCopilotStream({
      sessionId,
      hydratedMessages,
      hasActiveStream,
      refetchSession: sessionQuery.refetch,
      copilotMode: "fast",
      copilotModel: undefined,
    });

  const { queuedMessages, appendChip } = useCopilotPendingChips({
    sessionId,
    status,
    messages,
    setMessages,
  });

  // Track the currently-selected graph so the async bind effect can
  // discard stale responses.  Updated synchronously every render so the
  // IIFE sees the freshest value after `await`.
  const currentFlowIDRef = useRef<string | null>(flowID ?? null);
  useEffect(() => {
    currentFlowIDRef.current = flowID ?? null;
  }, [flowID]);

  const boundGraphRef = useRef<string | null>(null);
  // Declared here (before the reset effect) so the reset effect can clear
  // it on graph change.  Without this clear, a bind still in-flight when
  // the user switches graphs would leave ``bindingRef.current === true``
  // and the new graph's bind effect would early-return without ever
  // retrying — panel silently stuck bootstrapping. See sentry 13568553.
  const bindingRef = useRef(false);

  // Reset on graph change MUST run before the bind effect so that navigating
  // between agents first clears the old session/messages (same render cycle)
  // and only then the bind effect tries to create a new session.  Reverse
  // ordering leaks the previous graph's session id + messages into the new
  // graph for one paint.
  useEffect(() => {
    if (!flowID) {
      setSessionId(null);
      setRevertTargetVersion(null);
      setMessages([]);
      boundGraphRef.current = null;
      bindingRef.current = false;
      setBindError(null);
      return;
    }
    if (boundGraphRef.current && boundGraphRef.current !== flowID) {
      setSessionId(null);
      setRevertTargetVersion(null);
      setMessages([]);
      boundGraphRef.current = null;
      bindingRef.current = false;
      setBindError(null);
    }
  }, [flowID, setMessages]);

  // Bind the panel session to (flowID -> graph_id).  Navigating to a
  // different graph or clearing flowID drops the current session (above) so
  // the next panel open starts clean with the right graph.  Guards against:
  //   1) concurrent re-entry while an in-flight bind is pending
  //      (`bindingRef`) — rapid open/close toggles would otherwise fire
  //      multiple POST /sessions calls for the same graph.
  //   2) stale async responses after the user switches graphs
  //      (`currentFlowIDRef`) — an older graph's response must NOT
  //      overwrite a newer graph's sessionId.
  useEffect(() => {
    if (!isOpen) return;
    if (!flowID) return;
    if (boundGraphRef.current === flowID && sessionId) return;
    if (bindingRef.current) return;
    const effectFlowID = flowID;
    boundGraphRef.current = effectFlowID;
    bindingRef.current = true;
    setBindError(null);

    void (async () => {
      try {
        const response = (await createBuilderSession({
          data: { builder_graph_id: effectFlowID },
        })) as unknown as {
          status: number;
          data?: { id?: string };
        };
        // The user may have navigated to a different graph while we were
        // awaiting the response — in that case, discard this one.  The
        // reset effect above will have already cleared boundGraphRef; the
        // next render fires a fresh bind for the new flowID.
        if (currentFlowIDRef.current !== effectFlowID) {
          return;
        }
        if (response.status !== 200 || !response.data?.id) {
          throw new Error("failed_to_bind_builder_session");
        }
        setSessionId(response.data.id);
      } catch (err) {
        if (currentFlowIDRef.current !== effectFlowID) return;
        Sentry.captureException(err);
        setBindError("failed_to_bind_builder_session");
        // Clear the bound marker so the panel can re-trigger on retry.
        boundGraphRef.current = null;
        toast({
          variant: "destructive",
          title: "Could not start the builder chat",
          description: "Please retry or close and reopen the chat panel.",
        });
      } finally {
        bindingRef.current = false;
      }
    })();
    // `bindRetryToken` is intentionally in the dep array so retrying bumps
    // the token and forces this effect to re-run even when flowID+sessionId
    // haven't changed.
  }, [isOpen, flowID, sessionId, bindRetryToken, createBuilderSession, toast]);

  // Auto-create a blank agent when the panel is opened without one.
  // The saved graph's id becomes the builder session's binding.  On
  // failure we surface a retry button (see `bootstrapError` in the
  // return value) so the user can recover without closing the panel.
  const isBootstrappingRef = useRef(false);
  useEffect(() => {
    if (!isOpen || flowID || isBootstrappingRef.current) return;
    isBootstrappingRef.current = true;
    setBootstrapError(null);
    void (async () => {
      try {
        const response = (await createNewGraph({
          data: {
            graph: {
              name: `New Agent ${new Date().toISOString()}`,
              description: "",
              nodes: [],
              links: [],
            },
            source: "builder",
          },
        })) as unknown as {
          status: number;
          data?: GraphModel;
        };
        if (response.status !== 200 || !response.data?.id) {
          throw new Error("failed_to_bootstrap_agent");
        }
        setQueryStates({
          flowID: response.data.id,
          flowVersion: response.data.version,
        });
      } catch (err) {
        Sentry.captureException(err);
        setBootstrapError("failed_to_bootstrap_agent");
        toast({
          variant: "destructive",
          title: "Could not create a blank agent",
          description: "Please try again.",
        });
      } finally {
        isBootstrappingRef.current = false;
      }
    })();
  }, [
    isOpen,
    flowID,
    bootstrapRetryToken,
    createNewGraph,
    setQueryStates,
    toast,
  ]);

  // Inline tool-integration: run_agent exec_id -> URL; edit_agent -> graph refetch + record revert point.
  const processedToolCallsRef = useRef(new Set<string>());
  useEffect(() => {
    processedToolCallsRef.current = new Set();
  }, [flowID]);

  const latestVersionBeforeEditRef = useRef<number | null>(null);
  useEffect(() => {
    // Capture the active version any time we load the graph.  This is the
    // version we "revert to" after the next edit_agent turn succeeds.
    if (graph?.version != null && !hasActiveStream) {
      latestVersionBeforeEditRef.current = graph.version;
    }
  }, [graph?.version, hasActiveStream]);

  // Process tool outputs as soon as they reach output-available — do NOT gate
  // on status === "ready". run_agent often completes mid-turn (followed by
  // more assistant text), and edit_agent can finish before the wrap-up
  // summary is streamed — gating on ready misses both.
  //
  // Tool parts use the AI SDK static-typed convention `tool-<name>` (NOT
  // `dynamic-tool` with a `toolName` field). Matching on part.type directly.
  //
  // IMPORTANT: on the live stream the backend emits `output` as a JSON
  // STRING (see backend copilot SDK response_adapter — tool outputs are
  // stashed as strings). After hydration from DB, `convertChatSessionToUi`
  // parses that string to an object. So this effect must handle BOTH shapes
  // to work on live-streamed *and* hydrated sessions.
  useEffect(() => {
    // Drop tool-parts from the previous graph's stream before the reset
    // effect flushes them — otherwise flowVersion / flowExecutionID get
    // written with stale values. `null` is the initial "no session yet"
    // state and must pass through so hydrated messages still apply.
    if (boundGraphRef.current !== null && boundGraphRef.current !== flowID) {
      return;
    }
    for (const msg of messages) {
      if (msg.role !== "assistant") continue;
      for (const part of msg.parts ?? []) {
        if (part.type !== "tool-edit_agent" && part.type !== "tool-run_agent") {
          continue;
        }
        const toolPart = part as {
          type: string;
          toolCallId: string;
          state: string;
          output?: unknown;
        };
        if (toolPart.state !== "output-available") continue;
        if (processedToolCallsRef.current.has(toolPart.toolCallId)) continue;

        const output = parseToolOutput(toolPart.output);
        // Only mark as processed once we successfully extract a usable
        // object — otherwise a mid-stream partial string would lock us out
        // of the real output that arrives milliseconds later.
        if (!output) continue;
        processedToolCallsRef.current.add(toolPart.toolCallId);

        if (part.type === "tool-edit_agent") {
          // Record the version we were on before this edit so the user can
          // roll back to it. If the tool returned the new graph_version,
          // switch the URL to that version so the builder canvas re-renders
          // the edited graph — otherwise the URL stays pinned to the old
          // version and refetchGraph returns the same data.
          //
          // Snapshot the pre-edit version synchronously and advance the ref
          // to the new version (if the tool returned one) so that a second
          // rapid edit captures the correct revert target — not the
          // pre-first-edit version which the async refetchGraph hasn't
          // updated yet.
          const preEditVersion = latestVersionBeforeEditRef.current;
          if (preEditVersion != null) {
            setRevertTargetVersion(preEditVersion);
          }
          const newVersion = output.graph_version;
          if (typeof newVersion === "number" && Number.isFinite(newVersion)) {
            latestVersionBeforeEditRef.current = newVersion;
            setQueryStates({ flowVersion: newVersion });
          }
          void refetchGraph();
          if (flowID) {
            queryClient.invalidateQueries({
              queryKey: getGetV1GetSpecificGraphQueryKey(flowID, {}),
            });
          }
        } else if (part.type === "tool-run_agent") {
          // run_agent's output can be either ExecutionStartedResponse
          // (async enqueue → execution_id on output directly) or
          // AgentOutputResponse for a sync wait_for_result path
          // (execution_id nested under output.execution).
          const direct = output.execution_id;
          const nested = (output.execution as Record<string, unknown> | null)
            ?.execution_id;
          const execId = typeof direct === "string" ? direct : nested;
          if (typeof execId === "string" && /^[\w-]+$/i.test(execId)) {
            setQueryStates({ flowExecutionID: execId });
          }
        }
      }
    }
  }, [messages, flowID, refetchGraph, queryClient, setQueryStates]);

  // Escape-to-close when the panel is focused.  Skip inside editable
  // elements so Escape does not discard an in-progress draft.
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

  function handleToggle() {
    setIsOpen((o) => !o);
  }

  async function handleRevert() {
    if (!flowID || revertTargetVersion == null) return;
    try {
      const response = (await setActiveVersion({
        graphId: flowID,
        data: { active_graph_version: revertTargetVersion },
      })) as unknown as { status: number };
      if (response.status !== 200) {
        throw new Error("failed_to_revert");
      }
      setQueryStates({ flowVersion: revertTargetVersion });
      await refetchGraph();
      queryClient.invalidateQueries({
        queryKey: getGetV1GetSpecificGraphQueryKey(flowID, {}),
      });
      if (sessionId) {
        queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(sessionId),
        });
      }
      setRevertTargetVersion(null);
      toast({
        title: "Reverted to the previous version",
        description: `Now viewing version ${revertTargetVersion}.`,
      });
    } catch (err) {
      Sentry.captureException(err);
      toast({
        variant: "destructive",
        title: "Revert failed",
        description: "Please try again.",
      });
    }
  }

  async function onSend(message: string, _files?: File[]) {
    const trimmed = message.trim();
    if (!trimmed) return;
    if (!sessionId) return;
    const isInFlight = status === "streaming" || status === "submitted";
    if (isInFlight) {
      appendChip(trimmed);
      try {
        const { queueFollowUpMessage } = await import(
          "@/app/(platform)/copilot/helpers/queueFollowUpMessage"
        );
        await queueFollowUpMessage(sessionId, trimmed);
      } catch (err) {
        Sentry.captureException(err);
        toast({
          variant: "destructive",
          title: "Could not queue message",
          description: "Please wait for the current response to finish.",
        });
      }
      return;
    }
    sendMessage({ text: trimmed });
  }

  // While an error is active the panel surfaces a retry button instead of
  // the loading spinner — so the computed bootstrapping flag must read
  // false in that case.  Without this, a bind / create-graph failure would
  // still render "Preparing builder chat…" forever.
  const isBootstrapping =
    !bindError &&
    !bootstrapError &&
    (isBootstrappingGraph ||
      (!flowID && isOpen) ||
      (isOpen && !!flowID && !sessionId));

  function retryBind() {
    setBindError(null);
    setBindRetryToken((t) => t + 1);
  }

  function retryBootstrap() {
    setBootstrapError(null);
    setBootstrapRetryToken((t) => t + 1);
  }

  return {
    isOpen,
    handleToggle,
    panelRef,
    sessionId,
    flowID: flowID ?? null,
    flowVersion: flowVersion ?? null,
    messages,
    status,
    error,
    stop,
    onSend,
    queuedMessages,
    isBootstrapping,
    revertTargetVersion,
    handleRevert,
    bindError,
    bootstrapError,
    retryBind,
    retryBootstrap,
  };
}
