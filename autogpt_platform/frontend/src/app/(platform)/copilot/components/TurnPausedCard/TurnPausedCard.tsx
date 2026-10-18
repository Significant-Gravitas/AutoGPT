"use client";

import { LivePausedCard } from "./components/LivePausedCard/LivePausedCard";
import { PausedLine } from "./components/PausedLine";
import { pauseTitle } from "./helpers";
import { useTurnPausedCard } from "./useTurnPausedCard";

interface Props {
  data: unknown;
  pauseKey: string;
  /** The pause is a stored row, so its key outlives a reload. */
  isStoredRow: boolean;
}

/**
 * Where a turn that landed at a limit ends: what stopped it and the ways to
 * pick it back up. Only the pause the chat is still sitting on offers them;
 * any other reads as a one-line record of what happened.
 */
export function TurnPausedCard({ data, pauseKey, isStoredRow }: Props) {
  const card = useTurnPausedCard({ data, pauseKey, isStoredRow });
  if (!card.pause) return null;
  if (card.isLive && card.sessionId) {
    return (
      <LivePausedCard
        pause={card.pause}
        sessionId={card.sessionId}
        assistantName={card.assistantName}
        onFollow={card.followTurn}
        onStop={card.stop}
      />
    );
  }
  return <PausedLine title={pauseTitle(card.pause)} note={card.note} />;
}
