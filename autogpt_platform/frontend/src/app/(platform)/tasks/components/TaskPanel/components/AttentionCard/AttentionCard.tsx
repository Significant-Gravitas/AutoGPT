"use client";

import type { Ask } from "@/app/api/__generated__/models/ask";
import type { Blocker } from "@/app/api/__generated__/models/blocker";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { FormEvent } from "react";
import { actionLabel } from "../../../../data/helpers";
import { orderOptions } from "./helpers";
import { useAttentionCard } from "./useAttentionCard";

interface Props {
  taskId: string;
  goal: string;
  askerName: string;
  ask: Ask | null;
  blocker: Blocker | null;
  awaitingAccept: boolean;
  resultSummary?: string | null;
  unmet?: string[];
}

const INPUT =
  "h-8 w-full min-w-0 flex-1 rounded-control bg-surface px-2.5 text-body-sm text-foreground-ink shadow-card transition-shadow duration-150 ease-out-expo placeholder:text-foreground-faint focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50";

/** What the task needs from you, pinned under the goal: an open question,
 *  what stopped it, or finished work waiting for your OK. */
export function AttentionCard({
  taskId,
  goal,
  askerName,
  ask,
  blocker,
  awaitingAccept,
  resultSummary,
  unmet = [],
}: Props) {
  const card = useAttentionCard(taskId, goal);

  if (!ask && !blocker && !awaitingAccept) return null;

  function handleAnswer(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    card.sendAnswer(card.answer, ask?.id);
  }

  function handleGoal(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    card.submitGoal();
  }

  const stuck = !ask && !!blocker;
  const options = ask ? orderOptions(ask) : [];
  const actions = ask
    ? []
    : blocker
      ? (blocker.next_actions ?? [])
      : ["accept"];

  return (
    <section
      aria-label={stuck ? "What stopped it" : "Needs you"}
      className={cn(
        "flex animate-inbox-enter flex-col gap-3 rounded-card p-4",
        stuck ? "bg-warning-surface" : "bg-surface-wash",
      )}
    >
      <div className="flex items-center gap-2">
        <span
          aria-hidden
          className={cn(
            "shrink-0 rounded-full",
            stuck ? "size-1.5 bg-warning" : "size-2 bg-status-waiting",
          )}
        />
        <Text variant="inbox-medium" tone={stuck ? "warning" : "waiting"}>
          {stuck ? "Stuck" : "Needs you"}
        </Text>
        {ask ? (
          <Text variant="inbox" tone="quiet">
            · {askerName} asks
          </Text>
        ) : null}
      </div>

      {ask ? (
        <>
          <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
            {ask.question}
          </Text>
          {ask.why ? (
            <Text variant="small" tone="quiet">
              {ask.why}
            </Text>
          ) : null}
          {options.length ? (
            <div className="flex flex-wrap gap-1.5">
              {options.map((option) => (
                <Button
                  key={option}
                  variant={
                    option === ask.default_option ? "primary" : "secondary"
                  }
                  size="xs"
                  className="max-w-full"
                  title={option}
                  disabled={card.isAnswering}
                  loading={card.pendingAnswer === option}
                  onClick={() => card.sendAnswer(option, ask.id)}
                >
                  <span className="truncate">{option}</span>
                </Button>
              ))}
            </div>
          ) : null}
          <form onSubmit={handleAnswer} className="flex items-center gap-2">
            <input
              type="text"
              aria-label="Your answer"
              placeholder={
                options.length ? "Or type your own" : "Type your answer"
              }
              className={INPUT}
              value={card.answer}
              onChange={(event) => card.setAnswer(event.target.value)}
              disabled={card.isAnswering}
            />
            <Button
              type="submit"
              variant={options.length ? "secondary" : "primary"}
              size="xs"
              loading={
                card.isAnswering && card.pendingAnswer === card.answer.trim()
              }
              disabled={!card.answer.trim()}
            >
              Send
            </Button>
          </form>
        </>
      ) : blocker ? (
        <>
          <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
            {blocker.what_stopped_it}
          </Text>
          {blocker.done_so_far ? (
            <Text variant="small" tone="quiet">
              So far: {blocker.done_so_far}
            </Text>
          ) : null}
        </>
      ) : (
        <>
          <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
            {resultSummary || "The work is finished and waiting for your OK."}
          </Text>
          {unmet.length ? (
            <Text variant="small" tone="warning">
              Not met: {unmet.join("; ")}
            </Text>
          ) : null}
        </>
      )}

      {actions.length && !card.isEditingGoal ? (
        <div className="flex flex-wrap gap-1.5">
          {actions.map((action, index) => (
            <Button
              key={action}
              variant={index === 0 ? "primary" : "secondary"}
              size="xs"
              className="max-w-full"
              title={actionLabel(action)}
              disabled={card.isActing}
              loading={card.pendingAction === action}
              onClick={() => card.runAction(action)}
            >
              <span className="truncate">{actionLabel(action)}</span>
            </Button>
          ))}
        </div>
      ) : null}

      {card.isEditingGoal ? (
        <form onSubmit={handleGoal} className="flex flex-col gap-2">
          <input
            type="text"
            autoFocus
            aria-label="New goal"
            className={INPUT}
            value={card.newGoal}
            onChange={(event) => card.setNewGoal(event.target.value)}
            disabled={card.isActing}
          />
          <div className="flex gap-1.5">
            <Button
              type="submit"
              variant="primary"
              size="xs"
              loading={card.isActing}
              disabled={!card.newGoal.trim()}
            >
              Change goal
            </Button>
            <Button
              type="button"
              variant="ghost-muted"
              size="xs"
              onClick={card.cancelGoal}
            >
              Cancel
            </Button>
          </div>
        </form>
      ) : null}

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
