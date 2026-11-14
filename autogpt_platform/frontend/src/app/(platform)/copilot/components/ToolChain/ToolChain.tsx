"use client";

import { useStableCallback } from "@/hooks/useStableCallback";
import {
  Alert02Icon,
  ArrowDown01Icon,
  SentIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { domAnimation, LazyMotion, m, useReducedMotion } from "framer-motion";
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
import { toastSendFailure } from "../../helpers/sendFailureToast";
import { useCopilotChatActions } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { AnsweredQuestions } from "../AnsweredQuestions/AnsweredQuestions";
import { ChainActionCard } from "../ChainActionCard/ChainActionCard";
import { AnsweredQuestionsContext } from "../QuestionDock/AnsweredQuestionsContext";
import { PendingQuestionsContext } from "../QuestionDock/PendingQuestionsContext";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import { ACCORDION_PANEL, accordionState, PANEL_REVEAL } from "./accordion";
import {
  buildChainReply,
  ChainActionsContext,
  type ChainActionEntry,
  uniqueChainEntries,
} from "./chainActions";
import { useCredentialFailureCounters } from "./useCredentialFailureCounters";
import { HeldOutcomesContext } from "../ChatMessagesContainer/HeldOutcomesContext";
import { ChainRows } from "./ChainRows";
import {
  formatToolDuration,
  totalToolDuration,
} from "../../helpers/toolDurations";
import { useNoteRunningSteps } from "./StepClock";
import { applyHeldOutcome } from "./heldRow";
import { useHeldAnswersStore } from "../ApprovalQueue/heldAnswersStore";
import {
  type ChainRow,
  getAskedQuestions,
  getChainHeading,
  isLiftedSetupRow,
  isToolCallPending,
  markSupersededSubSessionRows,
  toChainRow,
} from "./helpers";
import { SwapText } from "./SwapText";
import { ToolResult } from "./ToolResult";
import { startedTaskOf } from "../TaskStatusLine/helpers";
import { TaskStatusLine } from "../TaskStatusLine/TaskStatusLine";

const COLLAPSED_WINDOW = 2;
// A burst of fast tools changes the live heading faster than it can be read:
// each heading stays up at least this long, then the newest one shows.
const HEADING_HOLD_MS = 500;
const EASE_OUT_QUINT = [0.22, 1, 0.36, 1] as const;

async function sendOnce(
  sending: { current: boolean },
  entries: ChainActionEntry[],
  message: string,
  sendReply: (message: string) => void,
) {
  if (sending.current) return;
  sending.current = true;
  try {
    await Promise.all(entries.map((entry) => entry.beforeSend?.()));
    entries.forEach((entry) => entry.onSent?.());
    sendReply(message);
  } finally {
    sending.current = false;
  }
}

interface Props {
  parts: MessagePart[];
  isStreaming: boolean;
  /** Public share viewer: the chain renders as the owner saw it, but setup
   *  cards and Proceed are the owner's work — a reader gets no Connect
   *  prompt and no way to send a follow-up turn. */
  readOnly?: boolean;
}

export function ToolChain({ parts, isStreaming, readOnly = false }: Props) {
  const [manualExpanded, setManualExpanded] = useState<boolean | null>(null);
  const panelId = useId();
  const reducedMotion = useReducedMotion();

  const pendingQuestions = useContext(PendingQuestionsContext);
  const answeredQuestions = useContext(AnsweredQuestionsContext);
  const heldOutcomes = useContext(HeldOutcomesContext);
  const heldAnswers = useHeldAnswersStore((state) => state.answers);
  const { onSend } = useCopilotChatActions();
  // The ref latches against a double effect run; the state re-renders Proceed.
  const autoSentRef = useRef(false);
  const sendingRef = useRef(false);
  const [autoSent, setAutoSent] = useState(false);

  // Action cards (credential setup, clarifying questions) register here
  // instead of rendering their own Proceed/Answer buttons — the chain
  // renders one Proceed that sends everything as a single message.
  // Keyed by the registering component, not the card id (see ChainActions).
  const [registrations, setRegistrations] = useState<
    ReadonlyMap<string, ChainActionEntry>
  >(new Map());
  // Kept as memos with the compiler on: the cards' register effects depend on
  // this context value, and re-running them would re-register every card.
  const register = useCallback((owner: string, entry: ChainActionEntry) => {
    setRegistrations((prev) => {
      const next = new Map(prev);
      next.set(owner, entry);
      return next;
    });
  }, []);
  const unregister = useCallback((owner: string) => {
    setRegistrations((prev) => {
      if (!prev.has(owner)) return prev;
      const next = new Map(prev);
      next.delete(owner);
      return next;
    });
  }, []);
  const chainActions = useMemo(
    () => ({ register, unregister }),
    [register, unregister],
  );

  const pendingActions = uniqueChainEntries(registrations);
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
  const needsManualProceed = pendingActions.some(
    (entry) => entry.manualProceed,
  );
  const hasCardWork =
    connectorRequests.length > 0 ||
    mcpRequests.length > 0 ||
    inputRequests.length > 0 ||
    questionRequests.length > 0;
  const allActionsReady =
    pendingActions.length > 0 && pendingActions.every((entry) => entry.ready);
  // One tool can need several accounts. The turn goes out once, when every
  // card in the chain is satisfied — otherwise connecting the first provider
  // sends while the rest of the rows are still visibly unconnected.
  const justConnectedHere = pendingActions.some((entry) => entry.justConnected);
  // A tool still awaiting its result may yet add a card to this chain, and the
  // send is once-only — spending it now leaves that card unsendable. A stopped
  // stream owes no more results, however its calls ended.
  const awaitingToolResult = isStreaming && parts.some(isToolCallPending);
  const canAutoSend =
    !readOnly &&
    !awaitingToolResult &&
    !autoSent &&
    allActionsReady &&
    !needsManualProceed &&
    justConnectedHere;
  // Nothing to sign in to and nothing to fill in: without a button this card
  // has no way forward at all — a card that lands after the send is spent
  // included. Only asks that build a message qualify — an MCP row builds none,
  // so offering one there is a button that does nothing.
  const offerProceed =
    !readOnly &&
    !awaitingToolResult &&
    !canAutoSend &&
    allActionsReady &&
    (connectorRequests.length > 0 || inputRequests.length > 0);

  // The cards are already gone by the time a send can fail — the user's
  // message is appended optimistically and the chat's error banner offers
  // Retry — so the failure only needs to be said out loud, and the toast
  // points at the thread rather than the composer.
  function sendReply(message: string) {
    void Promise.resolve(onSend(message)).catch((error: unknown) =>
      toastSendFailure({
        title: "Couldn't send message",
        error,
        recovery: "it is still in the thread, use Retry to send it again",
        paywallRecovery:
          "it is still in the thread, use Retry once you have a plan",
      }),
    );
  }

  // Proceed stays clickable while beforeSend runs, and a second click would
  // send the same reply again. The ref is set before the first await, so the
  // second click sees it in time.
  function sendAfter(entries: ChainActionEntry[], message: string) {
    return sendOnce(sendingRef, entries, message, sendReply);
  }

  const autoSendOnce = useStableCallback(() => {
    if (!canAutoSend || autoSentRef.current) return;
    autoSentRef.current = true;
    setAutoSent(true);
    const message = buildChainReply(pendingActions);
    if (!message) return;
    void sendAfter(pendingActions, message);
  });
  // pendingActions is rebuilt every render; the ref makes this once-per-chain.
  useEffect(
    function sendOnceEveryCardIsSatisfied() {
      autoSendOnce();
    },
    [canAutoSend],
  );

  useCredentialFailureCounters({ entries: registrations });

  // Kept as a memo: the compiler folds this into the scope of everything
  // rendered below the early return, so without it the rows would be rebuilt
  // whenever the chain expands, starts or stops streaming, or a card
  // registers.
  const rows = useMemo(
    () =>
      markSupersededSubSessionRows(
        parts
          .map((part, i) => toChainRow(part, i))
          .filter((row): row is ChainRow => row !== null)
          .map((row) => applyHeldOutcome(row, heldOutcomes, heldAnswers))
          // Unanswered clarifying questions and setup cards render their work
          // in the card below the chain — their rows are lifted out of view.
          .map((row) =>
            pendingQuestions?.callIds.includes(row.key)
              ? { ...row, requiresAction: true, lifted: true }
              : isLiftedSetupRow(row)
                ? { ...row, lifted: true }
                : row,
          ),
      ),
    [parts, pendingQuestions, heldOutcomes, heldAnswers],
  );
  useNoteRunningSteps(rows);
  if (rows.length === 0) return null;

  const askedRecords = rows.flatMap((row) => {
    const questions = getAskedQuestions(row);
    if (questions.length === 0) return [];
    const answers = answeredQuestions.get(row.key);
    return [
      {
        key: row.key,
        items: questions.map((q) => ({
          key: q.keyword,
          question: q.question,
          answer: answers?.get(q.keyword),
        })),
      },
    ];
  });
  const shownRows = rows.filter(
    (row) =>
      !row.lifted && !askedRecords.some((record) => record.key === row.key),
  );
  const liftedRows = rows.filter((row) => row.lifted);
  // A public transcript can't read the owner's tasks, so it shows none.
  const startedTasks = readOnly
    ? []
    : rows.flatMap((row) => {
        const task = row.state === "done" ? startedTaskOf(row.output) : null;
        return task ? [task] : [];
      });

  const expanded = manualExpanded === true;
  const heading = getChainHeading(rows, isStreaming && !expanded);
  // How long the chain's steps took together, once it has stopped.
  const chainTime = totalToolDuration(rows.map((row) => row.key));
  const hasError = rows.some((row) => row.state === "error");
  const hasRequiredAction = shownRows.some((row) => row.requiresAction);
  // Auto-open only while streaming; once the chain finishes (or on reload)
  // it collapses back to the heading, leaving just the action rows on
  // screen. A manual toggle overrides either direction and sticks.
  const open = manualExpanded ?? isStreaming;
  const windowMode = isStreaming && !expanded && !hasRequiredAction;
  // Action-required cards (credential setup etc.) stay on screen by default:
  // a chain that has collapsed on its own still shows its action rows. An
  // explicit collapse from the user closes them like any other row.
  const actionOnly = !open && hasRequiredAction && manualExpanded !== false;
  const panelOpen = open || actionOnly;
  // Rows stay mounted while closed so the 0fr collapse can animate — but
  // only the streaming window's rows. Mounting the full list the moment the
  // stream ends would flood the collapsing panel with entering rows; the
  // rest mount when the user actually expands.
  const visible = actionOnly
    ? shownRows.filter((row) => row.requiresAction)
    : windowMode || !panelOpen
      ? shownRows.slice(-COLLAPSED_WINDOW)
      : shownRows;

  // A finished chain closes with a "Done" step, so the rail always ends on
  // a resolved node instead of trailing off the last tool. An errored or
  // still-running chain has no such ending.
  const showDone = !isStreaming && !hasError && !windowMode && panelOpen;

  // Proceed sends the combined reply of every READY card as one message,
  // and their onSent callbacks fire at that moment. Unready cards (e.g. an
  // unconnected MCP server) are left out instead of blocking the ready ones.
  function handleProceed() {
    const readyActions = pendingActions.filter((entry) => entry.ready);
    const message = buildChainReply(readyActions);
    if (!message) return;
    void sendAfter(readyActions, message);
  }

  return (
    <LazyMotion features={domAnimation} strict>
      <div className="my-2">
        {shownRows.length > 0 && (
          // A chain that starts while the reply streams comes in as one
          // piece, header and first row together. Read at mount only, so
          // neither a re-render nor a reloaded transcript plays it again.
          <m.div
            initial={
              isStreaming && !reducedMotion ? { opacity: 0, y: 4 } : false
            }
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.2, ease: EASE_OUT_QUINT }}
          >
            <button
              type="button"
              onClick={() => setManualExpanded(!open)}
              aria-expanded={panelOpen}
              aria-controls={panelId}
              className="group/chain -mx-2 flex w-fit max-w-full items-center gap-1.5 rounded-lg px-2 py-1 text-left transition-colors duration-100 hover:bg-surface-hover"
            >
              <ChainGlyph
                state={isStreaming ? "running" : hasError ? "failed" : "done"}
              />
              <SwapText
                text={heading}
                shimmer={isStreaming && !expanded}
                minHoldMs={HEADING_HOLD_MS}
                className={
                  "min-w-0 text-body font-normal " +
                  (hasError && !isStreaming
                    ? "text-error"
                    : "text-foreground-body")
                }
              />
              {chainTime !== null && !isStreaming ? (
                <span className="shrink-0 text-body-sm text-foreground-faint tabular-nums">
                  · {formatToolDuration(chainTime)}
                </span>
              ) : null}
              <Icon
                icon={ArrowDown01Icon}
                size={12}
                className={
                  "shrink-0 text-foreground-subtle transition-transform duration-300 ease-out-quint " +
                  (open ? "rotate-180" : "")
                }
              />
            </button>
            <div className={ACCORDION_PANEL + " " + accordionState(panelOpen)}>
              <div
                id={panelId}
                aria-hidden={!panelOpen}
                inert={!panelOpen || undefined}
                className="min-h-0 overflow-hidden"
              >
                <div
                  className={
                    "flex flex-col pt-2.5 pl-0.5" +
                    (panelOpen && !windowMode ? " " + PANEL_REVEAL : "")
                  }
                >
                  <ChainRows
                    rows={visible}
                    isStreaming={isStreaming}
                    windowMode={windowMode}
                    showDone={showDone}
                    reducedMotion={!!reducedMotion}
                    newestLiveKey={isStreaming ? rows.at(-1)?.key : undefined}
                    chainActions={chainActions}
                    readOnly={readOnly}
                  />
                </div>
              </div>
            </div>
          </m.div>
        )}

        {/* Questions the user has moved past stay where their card was,
            with the answers given, instead of folding away with the chain. */}
        {askedRecords.length > 0 && (
          <div className="mt-1 flex flex-col gap-2">
            {askedRecords.map((record) => (
              <AnsweredQuestions key={record.key} items={record.items} />
            ))}
          </div>
        )}

        {/* Tasks started here keep a line under the chain, so work that
            runs in the background stays visible once the steps fold away. */}
        {startedTasks.length > 0 && (
          <ul className="mt-1 flex flex-col gap-2">
            {startedTasks.map((task) => (
              <li key={task.id}>
                <TaskStatusLine task={task} />
              </li>
            ))}
          </ul>
        )}

        {/* Lifted rows render off-screen: their cards must stay mounted so
            the ChainActionCard below keeps its registrations. A read-only
            transcript has no card to feed, so they stay unmounted. */}
        {!readOnly && liftedRows.length > 0 && (
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
        {!readOnly && hasCardWork && (
          <ChainActionCard
            connectors={connectorRequests}
            mcp={mcpRequests}
            inputs={inputRequests}
            questions={questionRequests}
            manualProceed={needsManualProceed || offerProceed}
            isReady={allActionsReady}
            onProceed={handleProceed}
          />
        )}

        {/* Cards with nothing to connect, fill in or answer (confirm-only)
            still need somewhere to send from. */}
        {!readOnly && pendingActions.length > 0 && !hasCardWork && (
          <div className="mt-1 flex flex-col items-start gap-2">
            <span className="flex items-center gap-1.5 text-body text-foreground-secondary">
              <Icon icon={SentIcon} size={16} tone="subtle" />
              {allActionsReady
                ? "Everything's filled in"
                : "Complete the steps above to continue"}
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

interface ChainGlyphProps {
  state: "running" | "done" | "failed";
}

// The small mark before the chain's line: turning while its steps run, a
// green tick once they are done, red when one failed.
function ChainGlyph({ state }: ChainGlyphProps) {
  if (state === "running") {
    return (
      <span
        aria-hidden
        className="size-3 shrink-0 animate-spin rounded-full border-2 border-foreground-faint border-t-transparent motion-reduce:animate-none"
      />
    );
  }
  return (
    <Icon
      icon={state === "failed" ? Alert02Icon : Tick02Icon}
      size={13}
      tone={state === "failed" ? "error" : "ok"}
      className="shrink-0"
      aria-hidden
    />
  );
}
