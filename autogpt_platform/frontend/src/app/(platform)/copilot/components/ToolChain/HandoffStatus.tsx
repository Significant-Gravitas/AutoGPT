"use client";

import { ArrowUpRight01Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  frozenStatusOf,
  type HandoffOutcome,
  type HandoffState,
  handoffStateOf,
  possessive,
} from "./expertHandoff";
import { HandoffRetry } from "./HandoffRetry";
import { StatusAvatar } from "./HandoffStatusAvatar";
import { useSubSessionEffectiveStatus } from "./SubSessionLive";

interface Props {
  outcome: HandoffOutcome;
  name: string;
  /** The hire; absent on outputs that don't carry it. */
  expertId: string | null;
  avatarUrl: string | null;
  color: string | null;
  /** Sends a chat message; absent on a read-only transcript. */
  onSend?: (message: string) => void;
}

/** Where the task went once the hire and hand-off landed. The confirm only
 *  queued it, so this follows the expert's thread from there. While the work
 *  is in flight it is a card; once it is over it recedes to one line. The
 *  hire itself needs no card of its own: the answered proposal above already
 *  records it. */
export function HandoffStatus({
  outcome,
  name,
  expertId,
  avatarUrl,
  color,
  onSend,
}: Props) {
  const live = useSubSessionEffectiveStatus(
    outcome.subSessionId,
    frozenStatusOf(outcome),
  );
  const state = handoffStateOf(outcome, live);
  const unknown = state === "unknown";
  const avatar = (size: number) => (
    <StatusAvatar
      name={name}
      avatarUrl={avatarUrl}
      color={color}
      state={state}
      size={size}
    />
  );

  if (state === "waiting" || state === "working" || unknown) {
    return (
      <div className="w-full overflow-hidden rounded-3xl border border-separator bg-surface">
        <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-3">
          {avatar(32)}
          <p className="min-w-[min(100%,14rem)] flex-1 text-sm text-foreground-secondary [overflow-wrap:anywhere]">
            <HandoffSentence outcome={outcome} name={name} state={state} />
          </p>
          {unknown && (
            <span className="inline-flex shrink-0 items-center rounded-full bg-surface-hover px-2 py-0.5 text-caption-sm font-medium leading-inherit text-foreground-tertiary">
              Status unknown
            </span>
          )}
          {outcome.link && (
            <NextLink
              href={outcome.link}
              className="inline-flex shrink-0 items-center gap-1.5 rounded-full border border-border-base bg-surface px-3 py-1.5 text-sm font-medium text-foreground-emphasis shadow-sm transition-colors hover:border-border-strong hover:bg-surface-muted focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus"
            >
              Open {possessive(name)} chat
              <Icon
                icon={ArrowUpRight01Icon}
                size={14}
                aria-hidden
                className="rtl:-scale-x-100"
              />
            </NextLink>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="flex items-start gap-2.5">
      {avatar(24)}
      <div className="flex min-w-0 flex-1 flex-wrap items-center gap-x-2.5 gap-y-1 pt-0.5">
        <p
          role={state === "failed" ? "status" : undefined}
          className="min-w-0 text-sm text-foreground-secondary [overflow-wrap:anywhere]"
        >
          <HandoffSentence outcome={outcome} name={name} state={state} />
        </p>
        {state === "failed"
          ? outcome.retry &&
            onSend &&
            expertId && (
              <HandoffRetry
                title={outcome.title}
                name={name}
                expertId={expertId}
                onSend={onSend}
              />
            )
          : outcome.link && (
              <NextLink
                href={outcome.link}
                className="inline-flex shrink-0 items-center gap-1 rounded text-sm font-medium text-foreground-emphasis underline-offset-2 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-focus"
              >
                Open {possessive(name)} chat
                <Icon
                  icon={ArrowUpRight01Icon}
                  size={14}
                  aria-hidden
                  className="rtl:-scale-x-100"
                />
              </NextLink>
            )}
      </div>
    </div>
  );
}

interface SentenceProps {
  outcome: HandoffOutcome;
  name: string;
  state: HandoffState;
}

function HandoffSentence({ outcome, name, state }: SentenceProps) {
  const who = (
    <span className="font-medium text-foreground-emphasis">{name}</span>
  );
  switch (state) {
    case "failed":
      return (
        <>
          {who} is hired, but {outcome.title} didn&apos;t start.
        </>
      );
    case "unconfirmed":
      return (
        <>
          {outcome.title} may have started in {possessive(name)} chat. Open it
          to check.
        </>
      );
    // An idle thread holds a result, a question for the user or an error;
    // the poll can't tell which, so the line doesn't guess.
    case "stopped":
      return (
        <>
          {who} stopped working on {outcome.title}. Their chat has the result,
          or a question for you.
        </>
      );
    case "waiting":
      return (
        <>
          {who} starts on {outcome.title} as soon as one of your other tasks
          finishes.
        </>
      );
    case "working":
      return (
        <>
          {who} is working on {outcome.title}.
        </>
      );
    // The live status can't be read: say only what is known.
    case "unknown":
      return (
        <>
          Sent {outcome.title} to {who}. Open their chat to see where it stands.
        </>
      );
  }
}
