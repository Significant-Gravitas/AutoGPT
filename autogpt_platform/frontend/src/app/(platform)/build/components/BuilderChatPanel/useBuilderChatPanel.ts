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
  usePostV2GetOrCreateBuilderSessionEndpoint,
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

export function useBuilderChatPanel({
  panelRef,
}: UseBuilderChatPanelArgs = {}) {
  const [isOpen, setIsOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [revertTargetVersion, setRevertTargetVersion] = useState<number | null>(
    null,
  );
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

  const { mutateAsync: createBuilderSession, isPending: isCreatingSession } =
    usePostV2GetOrCreateBuilderSessionEndpoint();
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

  // Bind the panel session to (flowID -> graph_id).  Navigating to a
  // different graph or clearing flowID drops the current session so the
  // next panel open starts clean with the right graph.
  const boundGraphRef = useRef<string | null>(null);
  useEffect(() => {
    if (!isOpen) return;
    if (!flowID) return;
    if (boundGraphRef.current === flowID && sessionId) return;
    boundGraphRef.current = flowID;

    void (async () => {
      try {
        const response = (await createBuilderSession({
          data: { graph_id: flowID },
        })) as unknown as {
          status: number;
          data?: { id?: string };
        };
        if (response.status !== 200 || !response.data?.id) {
          throw new Error("failed_to_bind_builder_session");
        }
        setSessionId(response.data.id);
      } catch (err) {
        Sentry.captureException(err);
        toast({
          variant: "destructive",
          title: "Could not start the builder chat",
          description: "Please try closing and reopening the chat panel.",
        });
      }
    })();
  }, [isOpen, flowID, sessionId, createBuilderSession, toast]);

  // Reset session + revert target when the graph changes (different agent).
  useEffect(() => {
    if (!flowID) {
      setSessionId(null);
      setRevertTargetVersion(null);
      setMessages([]);
      boundGraphRef.current = null;
      return;
    }
    if (boundGraphRef.current && boundGraphRef.current !== flowID) {
      setSessionId(null);
      setRevertTargetVersion(null);
      setMessages([]);
      boundGraphRef.current = null;
    }
  }, [flowID, setMessages]);

  // Auto-create a blank agent when the panel is opened without one.
  // The saved graph's id becomes the builder session's binding.
  const isBootstrappingRef = useRef(false);
  useEffect(() => {
    if (!isOpen || flowID || isBootstrappingRef.current) return;
    isBootstrappingRef.current = true;
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
        toast({
          variant: "destructive",
          title: "Could not create a blank agent",
          description: "Please try again.",
        });
      } finally {
        isBootstrappingRef.current = false;
      }
    })();
  }, [isOpen, flowID, createNewGraph, setQueryStates, toast]);

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
  useEffect(() => {
    for (const msg of messages) {
      if (msg.role !== "assistant") continue;
      for (const part of msg.parts ?? []) {
        if (typeof part.type === "string" && part.type.startsWith("tool-")) {
          // eslint-disable-next-line no-console
          console.debug(
            "[BuilderChatPanel] tool-part",
            part.type,
            (part as { state?: string }).state,
            (part as { toolCallId?: string }).toolCallId,
          );
        }
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
        processedToolCallsRef.current.add(toolPart.toolCallId);

        const output = toolPart.output as Record<string, unknown> | null;
        if (part.type === "tool-edit_agent") {
          // Record the version we were on before this edit so the user can
          // roll back to it. If the tool returned the new graph_version,
          // switch the URL to that version so the builder canvas re-renders
          // the edited graph — otherwise the URL stays pinned to the old
          // version and refetchGraph returns the same data.
          if (latestVersionBeforeEditRef.current != null) {
            setRevertTargetVersion(latestVersionBeforeEditRef.current);
          }
          // eslint-disable-next-line no-console
          console.debug("[BuilderChatPanel] edit_agent output", output);
          const newVersion = output?.graph_version;
          if (typeof newVersion === "number" && Number.isFinite(newVersion)) {
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
          const direct = output?.execution_id;
          const nested = (output?.execution as Record<string, unknown> | null)
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

  const isBootstrapping =
    isBootstrappingGraph ||
    (!flowID && isOpen) ||
    (isOpen && !!flowID && !sessionId);

  // The builder panel already auto-refetches the graph on edit_agent and
  // auto-opens the execution panel on run_agent (see effects above), so the
  // heavy tool-result cards those tools render (with "Open in library",
  // "Open in builder", "Open run" buttons) are redundant and confusing —
  // the user is already IN the builder looking at the updated agent / run.
  // Strip those tool parts from the rendered message list; the raw
  // `messages` above still drives the side-effects.
  // Earlier iterations hid edit_agent / run_agent / create_agent tool cards
  // on the assumption that the side-effect hooks (graph refetch, URL sub)
  // were enough feedback. They weren't — the user saw nothing in the chat
  // and couldn't tell if the tool actually ran. Render them as the shared
  // Copilot MessagePartRenderer does; the side effects still fire in the
  // effect above.
  const visibleMessages = messages;

  return {
    isOpen,
    handleToggle,
    panelRef,
    sessionId,
    flowID: flowID ?? null,
    flowVersion: flowVersion ?? null,
    messages: visibleMessages,
    status,
    error,
    stop,
    onSend,
    queuedMessages,
    isCreatingSession,
    isBootstrapping,
    revertTargetVersion,
    handleRevert,
  };
}
