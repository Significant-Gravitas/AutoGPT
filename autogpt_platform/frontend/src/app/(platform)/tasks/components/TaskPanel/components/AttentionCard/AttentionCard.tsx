"use client";

import type { Ask } from "@/app/api/__generated__/models/ask";
import type { Blocker } from "@/app/api/__generated__/models/blocker";
import type { CardApproval } from "@/app/api/__generated__/models/cardApproval";
import type { Evidence } from "@/app/api/__generated__/models/evidence";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CardApprovals } from "../../../CardApprovals/CardApprovals";
import { PanelMarkdown } from "../PanelMarkdown/PanelMarkdown";
import { ProofLine } from "../ProofLine/ProofLine";
import { AskControls } from "./components/AskControls";
import { CardButtons } from "./components/CardButtons";
import { CardLine } from "./components/CardLine";
import { GoalForm } from "./components/GoalForm";
import { attentionActions } from "./helpers";
import { useAttentionCard } from "./useAttentionCard";

interface Props {
  taskId: string;
  goal: string;
  askerName: string;
  /** The task failed: red, and Retry is always offered. */
  failed: boolean;
  ask: Ask | null;
  blocker: Blocker | null;
  approvals?: CardApproval[];
  awaitingAccept: boolean;
  resultSummary?: string | null;
  unmet?: string[];
  /** Proof for finished work waiting on your OK. */
  proof?: Evidence[];
  /** The work chat, where a file can be attached. */
  workChatHref: string | null;
}

/** The one card at the top of the panel when a task failed or waits on you:
 *  why, what it needs, and the buttons that answer it. */
export function AttentionCard({
  taskId,
  goal,
  askerName,
  failed,
  ask,
  blocker,
  approvals = [],
  awaitingAccept,
  resultSummary,
  unmet = [],
  proof = [],
  workChatHref,
}: Props) {
  const card = useAttentionCard(taskId, goal);
  const hasCards = approvals.length > 0;
  const showAsk = !!ask && !hasCards;
  const showResult = awaitingAccept && !ask && !blocker && !hasCards;

  if (!ask && !blocker && !awaitingAccept && !hasCards && !failed) return null;

  const why = blocker?.what_stopped_it || blocker?.short || ask?.why;
  const need = hasCards
    ? "Your OK on the cards below."
    : (ask?.question ?? (showResult ? "Your OK on this work." : null));
  const { actions, wantsUpload } = attentionActions({
    failed,
    blocker,
    waitsOnYou: showAsk || hasCards,
    awaitingAccept,
  });

  return (
    <section
      aria-label={failed ? "Failed" : "Needs you"}
      className={cn(
        "flex animate-inbox-enter flex-col gap-3 rounded-card p-4",
        failed ? "bg-error-surface" : "bg-surface-wash",
      )}
    >
      <div className="flex flex-wrap items-center gap-x-2">
        <span
          aria-hidden
          className={cn(
            "size-2 shrink-0 rounded-full",
            failed ? "bg-error" : "bg-status-waiting",
          )}
        />
        <Text variant="inbox-medium" tone={failed ? "error-strong" : "waiting"}>
          {failed ? "Failed" : "Needs you"}
        </Text>
        {showAsk || hasCards ? (
          <Text variant="inbox" tone="quiet">
            · {askerName} {hasCards ? "needs your OK" : "asks"}
          </Text>
        ) : null}
      </div>

      {showResult ? (
        <div className="flex flex-col gap-2">
          <PanelMarkdown>
            {resultSummary || "The work is finished and waiting for your OK."}
          </PanelMarkdown>
          {unmet.length ? (
            <Text variant="small" tone="warning">
              Not met: {unmet.join("; ")}
            </Text>
          ) : null}
          {proof.map((evidence) => (
            <ProofLine key={evidence.id} taskId={taskId} evidence={evidence} />
          ))}
        </div>
      ) : null}

      {why ? <CardLine label="Why">{why}</CardLine> : null}
      {blocker?.done_so_far ? (
        <Text variant="small" tone="quiet" className="break-words">
          So far: {blocker.done_so_far}
        </Text>
      ) : null}
      {need ? <CardLine label="What I need">{need}</CardLine> : null}

      {hasCards ? <CardApprovals taskId={taskId} cards={approvals} /> : null}
      {showAsk && ask ? (
        <AskControls
          ask={ask}
          answer={card.answer}
          onAnswerChange={card.setAnswer}
          onSend={card.sendAnswer}
          isAnswering={card.isAnswering}
          pendingAnswer={card.pendingAnswer}
        />
      ) : null}

      {card.isEditingGoal ? (
        <GoalForm
          value={card.newGoal}
          onChange={card.setNewGoal}
          onSubmit={card.submitGoal}
          onCancel={card.cancelGoal}
          isActing={card.isActing}
        />
      ) : (
        <CardButtons
          actions={actions}
          uploadHref={wantsUpload ? workChatHref : null}
          onAction={card.runAction}
          isActing={card.isActing}
          pendingAction={card.pendingAction}
        />
      )}

      {card.error ? (
        <Text variant="small" tone="error">
          {card.error instanceof Error && card.error.message
            ? card.error.message
            : "That didn't go through. Try again."}
        </Text>
      ) : null}
    </section>
  );
}
