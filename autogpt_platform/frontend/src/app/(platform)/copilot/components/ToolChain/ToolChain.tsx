"use client";

import {
  ArrowDown01Icon,
  SentIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import {
  AnimatePresence,
  domAnimation,
  LazyMotion,
  m,
  useReducedMotion,
} from "framer-motion";
import {
  useCallback,
  useContext,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
} from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { ChainActionCard } from "../ChainActionCard/ChainActionCard";
import { PendingQuestionsContext } from "../QuestionDock/PendingQuestionsContext";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import { ChainActionsContext, type ChainActionEntry } from "./chainActions";
import { ChainRowView } from "./ChainRowView";
import {
  type ChainRow,
  getChainHeading,
  isLiftedSetupRow,
  toChainRow,
} from "./helpers";
import { SwapText } from "./SwapText";
import { ToolResult } from "./ToolResult";

const COLLAPSED_WINDOW = 2;

interface Props {
  parts: MessagePart[];
  isStreaming: boolean;
}

export function ToolChain({ parts, isStreaming }: Props) {
  const [manualExpanded, setManualExpanded] = useState<boolean | null>(null);
  const panelId = useId();
  const reducedMotion = useReducedMotion();

  const pendingQuestions = useContext(PendingQuestionsContext);
  const { setInitialPrompt, sentMessageCount } = useCopilotUIStore();
  // Ids drafted by the last Proceed, plus the send count at that moment.
  // Proceed only fills the composer, so the cards' onSent callbacks fire
  // when the user actually sends — not when the draft is written.
  const draftedRef = useRef<{ ids: string[]; sentAt: number } | null>(null);

  // Action cards (credential setup, clarifying questions) register here
  // instead of rendering their own Proceed/Answer buttons — the chain
  // renders one Proceed that drafts everything into the chat input at once.
  const [actionEntries, setActionEntries] = useState<
    ReadonlyMap<string, ChainActionEntry>
  >(new Map());
  const register = useCallback((entry: ChainActionEntry) => {
    setActionEntries((prev) => {
      const next = new Map(prev);
      next.set(entry.id, entry);
      return next;
    });
  }, []);
  const unregister = useCallback((id: string) => {
    setActionEntries((prev) => {
      if (!prev.has(id)) return prev;
      const next = new Map(prev);
      next.delete(id);
      return next;
    });
  }, []);
  const chainActions = useMemo(
    () => ({ register, unregister }),
    [register, unregister],
  );

  useEffect(
    function notifyDraftedCardsOnSend() {
      const drafted = draftedRef.current;
      if (!drafted || sentMessageCount <= drafted.sentAt) return;
      draftedRef.current = null;
      drafted.ids.forEach((id) => actionEntries.get(id)?.onSent?.());
    },
    [sentMessageCount, actionEntries],
  );

  const rows = useMemo(
    () =>
      parts
        .map((part, i) => toChainRow(part, i))
        .filter((row): row is ChainRow => row !== null)
        // Unanswered clarifying questions and setup cards render their work
        // in the card below the chain — their rows are lifted out of view.
        .map((row) =>
          pendingQuestions?.callIds.includes(row.key)
            ? { ...row, requiresAction: true, lifted: true }
            : isLiftedSetupRow(row)
              ? { ...row, lifted: true }
              : row,
        ),
    [parts, pendingQuestions],
  );
  if (rows.length === 0) return null;

  const shownRows = rows.filter((row) => !row.lifted);
  const liftedRows = rows.filter((row) => row.lifted);

  const expanded = manualExpanded === true;
  const heading = getChainHeading(rows, isStreaming && !expanded);
  const hasError = rows.some((row) => row.state === "error");
  const hasRequiredAction = shownRows.some((row) => row.requiresAction);
  // Auto-open only while streaming; once the chain finishes (or on reload)
  // it collapses back to the heading, leaving just the action rows on
  // screen. A manual toggle overrides either direction and sticks.
  const open = manualExpanded ?? isStreaming;
  const windowMode = isStreaming && !expanded && !hasRequiredAction;
  // Action-required cards (credential setup etc.) can never leave the
  // screen: collapsing the chain hides the other rows but keeps the
  // action rows visible, and the streaming window is disabled for them.
  const actionOnly = !open && hasRequiredAction;
  const panelOpen = open || actionOnly;
  // Rows stay mounted while closed so the 0fr collapse can animate.
  const visible = actionOnly
    ? shownRows.filter((row) => row.requiresAction)
    : windowMode
      ? shownRows.slice(-COLLAPSED_WINDOW)
      : shownRows;

  // A finished chain closes with a "Done" step, so the rail always ends on
  // a resolved node instead of trailing off the last tool. An errored or
  // still-running chain has no such ending.
  const showDone = !isStreaming && !hasError && !windowMode;

  const pendingActions = [...actionEntries.values()];
  const connectorRequests = pendingActions
    .map((entry) => entry.connectors)
    .filter((request) => request !== undefined);
  const mcpRequests = pendingActions
    .map((entry) => entry.mcp)
    .filter((request) => request !== undefined);
  const inputRequests = pendingActions
    .map((entry) => entry.inputs)
    .filter((request) => request !== undefined);
  const questionRequests = pendingActions
    .map((entry) => entry.questions)
    .filter((request) => request !== undefined);
  const hasCardWork =
    connectorRequests.length > 0 ||
    mcpRequests.length > 0 ||
    inputRequests.length > 0 ||
    questionRequests.length > 0;
  const allActionsReady =
    pendingActions.length > 0 && pendingActions.every((entry) => entry.ready);

  // Proceed never sends: it drafts the combined reply of every READY card
  // into the chat input so the user reviews/edits and presses send
  // themselves. Unready cards (e.g. an unconnected MCP server) are left
  // out instead of blocking the ready ones. Cards stay registered until
  // the message actually goes out, at which point their onSent fires.
  function handleProceed() {
    const readyActions = pendingActions.filter((entry) => entry.ready);
    const message = readyActions
      .map((entry) => entry.buildMessage())
      .filter(Boolean)
      .join("\n\n");
    if (!message) return;
    draftedRef.current = {
      ids: readyActions.map((entry) => entry.id),
      sentAt: sentMessageCount,
    };
    setInitialPrompt(message);
  }

  return (
    <LazyMotion features={domAnimation} strict>
      <div className="my-2">
        {shownRows.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => setManualExpanded(!open)}
              aria-expanded={panelOpen}
              aria-controls={panelId}
              className="group/chain -mx-2 flex w-fit max-w-full items-center gap-1.5 rounded-lg px-2 py-1 text-left transition-colors duration-100 hover:bg-zinc-100"
            >
              <SwapText
                text={heading}
                shimmer={isStreaming && !expanded}
                className={
                  "min-w-0 text-sm font-normal " +
                  (hasError && !isStreaming ? "text-red-500" : "text-zinc-700")
                }
              />
              <Icon
                icon={ArrowDown01Icon}
                size={12}
                className={
                  "shrink-0 text-zinc-400 transition-transform duration-300 ease-out-quint " +
                  (open ? "rotate-180" : "")
                }
              />
            </button>
            <div className={ACCORDION_PANEL + " " + accordionState(panelOpen)}>
              <div
                id={panelId}
                aria-hidden={!panelOpen}
                inert={panelOpen ? undefined : ("" as unknown as boolean)}
                className="min-h-0 overflow-hidden"
              >
                <div
                  className={
                    "flex flex-col pl-0.5 pt-2.5" +
                    (panelOpen && !windowMode ? " " + PANEL_REVEAL : "")
                  }
                >
                  <AnimatePresence mode="popLayout">
                    {visible.map((row, i) => (
                      <m.div
                        key={row.key}
                        layout={!reducedMotion}
                        initial={
                          reducedMotion
                            ? false
                            : { opacity: 0, y: 8, scale: 0.985 }
                        }
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={
                          reducedMotion
                            ? undefined
                            : { opacity: 0, y: -6, scale: 0.985 }
                        }
                        transition={{
                          opacity: {
                            duration: reducedMotion ? 0 : 0.18,
                            delay: reducedMotion ? 0 : Math.min(i, 6) * 0.035,
                            ease: [0.22, 1, 0.36, 1],
                          },
                          y: {
                            duration: reducedMotion ? 0 : 0.22,
                            delay: reducedMotion ? 0 : Math.min(i, 6) * 0.035,
                            ease: [0.22, 1, 0.36, 1],
                          },
                          scale: {
                            duration: reducedMotion ? 0 : 0.22,
                            delay: reducedMotion ? 0 : Math.min(i, 6) * 0.035,
                            ease: [0.22, 1, 0.36, 1],
                          },
                          layout: {
                            duration: reducedMotion ? 0 : 0.22,
                            ease: [0.22, 1, 0.36, 1],
                          },
                        }}
                      >
                        <ChainActionsContext.Provider value={chainActions}>
                          <ChainRowView
                            row={row}
                            isLast={i === visible.length - 1 && !showDone}
                          />
                        </ChainActionsContext.Provider>
                      </m.div>
                    ))}
                  </AnimatePresence>
                  {showDone && (
                    <div className="flex items-stretch gap-2.5">
                      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-zinc-100">
                        <Icon
                          icon={Tick02Icon}
                          size={14}
                          className="text-zinc-600"
                        />
                      </div>
                      <div className="flex h-7 items-center text-sm text-zinc-600">
                        Done
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        )}

        {/* Lifted rows render off-screen: their cards must stay mounted so
            the ChainActionCard below keeps its registrations. */}
        {liftedRows.length > 0 && (
          <div className="hidden">
            <ChainActionsContext.Provider value={chainActions}>
              {liftedRows.map((row) => (
                <ToolResult key={row.key} row={row} />
              ))}
            </ChainActionsContext.Provider>
          </div>
        )}

        {/* Lifted out of the rows: connecting, filling inputs and answering
            questions are the user's own work, so every card's asks merge into
            one card under the chain that stays put when it collapses. */}
        {hasCardWork && (
          <ChainActionCard
            connectors={connectorRequests}
            mcp={mcpRequests}
            inputs={inputRequests}
            questions={questionRequests}
            isReady={allActionsReady}
            onProceed={handleProceed}
          />
        )}

        {/* Cards with nothing to connect, fill in or answer (confirm-only)
            still need somewhere to send from. */}
        {pendingActions.length > 0 && !hasCardWork && (
          <div className="mt-1 flex flex-col items-start gap-2">
            <span className="flex items-center gap-1.5 text-sm text-zinc-600">
              <Icon icon={SentIcon} size={16} className="text-zinc-400" />
              {allActionsReady
                ? "Everything's filled in — send it to continue"
                : "Complete the steps above, then send to continue"}
            </span>
            <Button
              variant="primary"
              size="small"
              disabled={!allActionsReady}
              onClick={handleProceed}
            >
              Proceed
            </Button>
          </div>
        )}
      </div>
    </LazyMotion>
  );
}
