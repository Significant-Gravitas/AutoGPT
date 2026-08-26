"use client";

import {
  collectCurrentTurn,
  getLiveNotice,
  isSessionLive,
  toMiniRow,
} from "./helpers";
import { LiveNotice, LiveSteps } from "./LiveSteps";
import { useLiveSubSession } from "./useLiveSubSession";

interface Props {
  subSessionId: string;
  active: boolean;
}

const MAX_STEPS = 3;

/** The delegate's work, live inside the parent chain: its recent tool calls
 *  and latest words, polled while the sub-session runs so the user watches
 *  progress without opening the session. The parent tool output stays
 *  ``running`` forever, so liveness comes from the polled session itself —
 *  once it goes idle, polling stops and the last state stays up as the
 *  delegate's final answer. */
export function SubSessionLive({ subSessionId, active }: Props) {
  const { session, isError, isPaused } = useLiveSubSession(
    subSessionId,
    active,
  );
  if (!active) return null;

  const turn = session ? collectCurrentTurn(session) : null;
  const recent = turn?.steps.slice(-MAX_STEPS) ?? [];
  const latestText = turn?.latestText ?? null;
  const notice = getLiveNotice({ isError, isPaused });
  if (!notice && recent.length === 0 && !latestText) return null;

  const isLive = !!session && isSessionLive(session) && !isPaused && !isError;
  const rows = recent.map((step, i) =>
    toMiniRow(step, i, isLive && i === recent.length - 1),
  );

  return (
    <div className="mt-2 border-t border-zinc-100 pl-1 pt-2.5">
      <LiveSteps rows={rows} />
      {latestText && (
        <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-zinc-600">
          {latestText}
        </p>
      )}
      {notice && <LiveNotice text={notice} subSessionId={subSessionId} />}
    </div>
  );
}
