"use client";

import {
  useGetV2GetSession,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import type { ToolUIPart } from "ai";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  getAnimationText,
  getToolCategory,
} from "../../tools/GenericTool/helpers";
import { useExpertMap } from "../../useExpertMap";
import type { ChainRow } from "./helpers";
import { CARD, StatusPill } from "./ResultCards";
import { asObject, str } from "./resultHelpers";
import { RowIcon } from "./RowIcon";
import { SwapText } from "./SwapText";
import { getCatalogLabel } from "./toolCatalog";

interface Props {
  subSessionId: string;
  active: boolean;
}

const POLL_MS = 3000;
const MAX_STEPS = 3;
// Live polling stops after this long on screen — a marathon sub-session is
// better followed via its "Open sub-session" link than by pinging the API
// every few seconds indefinitely. The card keeps its last snapshot.
const POLL_CAP_MS = 5 * 60_000;

interface LiveStep {
  name: string;
  input: unknown;
}

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

/** The polled sub-session plus the two states that used to render as
 *  nothing at all: a failed fetch, and the poll cap expiring while the run
 *  is still live. Both stop the polling, so both have to be visible —
 *  otherwise a dead card is indistinguishable from a working one. */
function useLiveSubSession(subSessionId: string, active: boolean) {
  const [isCapped, setIsCapped] = useState(false);
  useEffect(
    function stopPollingAfterCap() {
      if (!active) return;
      const timer = setTimeout(() => setIsCapped(true), POLL_CAP_MS);
      return () => clearTimeout(timer);
    },
    [active],
  );
  const { data, isError } = useGetV2GetSession(subSessionId, undefined, {
    query: {
      enabled: active && !!subSessionId,
      refetchInterval: (query) => {
        if (query.state.status === "error" || isCapped) return false;
        const raw = query.state.data;
        const polled = raw && raw.status === 200 ? raw.data : null;
        return !polled || isSessionLive(polled) ? POLL_MS : false;
      },
      select: (res) => (res.status === 200 ? res.data : null),
    },
  });
  const session = data ?? null;
  return {
    session,
    isError,
    // A capped poll on a finished session is not "paused" — the last
    // snapshot IS the final answer, so there is nothing left to watch.
    isPaused: isCapped && (!session || isSessionLive(session)),
  };
}

function getLiveNotice({
  isError,
  isPaused,
}: {
  isError: boolean;
  isPaused: boolean;
}) {
  if (isError) return "Couldn't load live updates";
  if (isPaused) return "Live updates paused";
  return null;
}

/** Says the polling stopped and keeps the deep link, so the user can follow
 *  the run at its source instead of watching a spinner that never resolves. */
function LiveNotice({
  text,
  subSessionId,
}: {
  text: string;
  subSessionId: string;
}) {
  return (
    <p className="mt-1.5 flex flex-wrap items-center gap-1.5 text-xs text-zinc-400">
      {text}
      <Link
        href={`/copilot?sessionId=${subSessionId}`}
        className="underline underline-offset-2 hover:text-zinc-600"
      >
        Open sub-session
      </Link>
    </p>
  );
}

function LiveSteps({ rows }: { rows: ChainRow[] }) {
  return rows.map((row, i) => {
    const isLastRow = i === rows.length - 1;
    return (
      <div key={row.key} className="flex items-stretch gap-2.5">
        <div className="flex w-6 flex-col items-center">
          <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-zinc-100">
            <RowIcon row={row} />
          </div>
          {!isLastRow && <div className="w-px flex-1 bg-zinc-200" />}
        </div>
        <div className={cn("min-w-0 flex-1 pt-[2px]", !isLastRow && "pb-2.5")}>
          <SwapText
            text={row.text}
            shimmer={row.state === "running"}
            className="max-w-full text-sm leading-5 text-zinc-600"
          />
        </div>
      </div>
    );
  });
}

/** A re-delegation reuses the same sub-session, so only the CURRENT turn
 *  (everything after the last user message) belongs to this card — the full
 *  history would replay the previous delegation's final answer here. */
function collectCurrentTurn(session: SessionDetailResponse) {
  const allMessages = Array.isArray(session.messages) ? session.messages : [];
  const lastUserIndex = allMessages.findLastIndex((m) => m.role === "user");
  const messages =
    lastUserIndex === -1 ? allMessages : allMessages.slice(lastUserIndex + 1);
  const steps: LiveStep[] = [];
  let latestText: string | null = null;
  for (const msg of messages) {
    const toolCalls = Array.isArray(msg.tool_calls) ? msg.tool_calls : [];
    for (const rawCall of toolCalls) {
      if (!rawCall || typeof rawCall !== "object") continue;
      const call = rawCall as {
        function?: { name?: unknown; arguments?: unknown };
      };
      const name = String(call.function?.name ?? "").trim();
      if (name)
        steps.push({ name, input: parseArguments(call.function?.arguments) });
    }
    if (
      msg.role === "assistant" &&
      typeof msg.content === "string" &&
      msg.content.trim()
    ) {
      latestText = msg.content.trim();
    }
  }
  return { steps, latestText };
}

/** Dress a polled tool call as a ChainRow so the delegate's steps reuse the
 *  main chain's icons and labels instead of raw tool names. */
function toMiniRow(step: LiveStep, index: number, running: boolean): ChainRow {
  const state = running ? "running" : "done";
  const catalog = getCatalogLabel(step.name, step.input, state);
  const category = catalog?.category ?? getToolCategory(step.name);
  const text =
    catalog?.text ??
    getAnimationText(
      {
        type: `tool-${step.name}`,
        state: running ? "input-available" : "output-available",
        input: step.input,
        toolCallId: `live-${index}`,
      } as ToolUIPart,
      getToolCategory(step.name),
    );
  return {
    key: `live-${step.name}-${index}`,
    category,
    text,
    state,
    tool: step.name,
    input: step.input,
  };
}

/** The card for a delegated run that hasn't returned yet. A blocking
 *  delegate/handoff yields no tool output while the teammate works, so
 *  everything comes from the tool INPUT: delegate/handoff carry expert_id +
 *  prompt, get_sub_session_result carries the exact sub_session_id. When
 *  only the expert is known, their currently-processing session is found by
 *  polling their session list; when only the session is known, the expert
 *  is read off the polled session itself. Swapped for the full
 *  SubSessionCard the moment the tool returns. */
interface PendingCardProps {
  input: unknown;
  minimal?: boolean;
}

export function SubSessionPendingCard({
  input,
  minimal = false,
}: PendingCardProps) {
  const { expertsById } = useExpertMap();
  const args = asObject(input) ?? {};
  const inputExpertId = str(args, "expert_id");
  const prompt = str(args, "prompt");
  const inputSessionId = str(args, "sub_session_id") ?? str(args, "session_id");
  const discoveredId = useDelegatedSessionId(
    inputSessionId ? null : inputExpertId,
  );
  const liveSessionId = inputSessionId ?? discoveredId;
  // Same query key as the live view below, so a full card polls once.
  const {
    session: liveSession,
    isError,
    isPaused,
  } = useLiveSubSession(liveSessionId ?? "", !!liveSessionId);
  const expertId = inputExpertId ?? liveSession?.expert_id ?? null;
  const expert = expertId ? expertsById.get(expertId) : undefined;

  return (
    <div className={cn(CARD, "w-full rounded-2xl p-2.5")}>
      <div className="flex items-center gap-2.5">
        <ExpertAvatar
          name={expert?.name ?? "Sub-AutoPilot"}
          avatarUrl={expert?.avatarUrl ?? null}
          size={28}
        />
        <p className="min-w-0 flex-1 truncate text-sm font-medium text-zinc-800">
          {expert?.name ?? "Sub-AutoPilot"}
          {expert?.role && (
            <span className="ml-1.5 font-normal text-zinc-400">
              {expert.role}
            </span>
          )}
        </p>
        {/* Once the poll dies the card no longer knows the run is going —
            keeping the spinner up would be a guess dressed as a fact. */}
        <StatusPill
          status={isError || isPaused ? "unknown" : "running"}
          className="text-sm"
        />
        {liveSessionId && (
          <Link
            href={`/copilot?sessionId=${liveSessionId}`}
            aria-label="Open sub-session"
            className="shrink-0 rounded-full p-1 text-zinc-400 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
          >
            <Icon icon={LinkSquare01Icon} size={14} />
          </Link>
        )}
      </div>
      {!minimal && prompt && (
        <p className="mt-1.5 line-clamp-2 pl-9 text-sm text-zinc-500">
          {prompt}
        </p>
      )}
      {!minimal && liveSessionId && (
        <SubSessionLive subSessionId={liveSessionId} active />
      )}
    </div>
  );
}

/** Find the delegated run's session while the tool output (which would name
 *  it) doesn't exist yet: the expert's currently-processing session. If the
 *  expert happens to be busy elsewhere this shows that work instead — still
 *  their live status, and the exact session takes over once the tool
 *  returns. */
function useDelegatedSessionId(expertId: string | null) {
  const mountedAtRef = useRef(Date.now());
  const { data } = useGetV2ListSessions(
    // Strict recency: the default puts pinned sessions first, so an expert
    // with a handful of pinned threads would push the running one out of
    // the window and the live view would never find it.
    { expert_id: expertId ?? undefined, limit: 5, pinned_first: false },
    {
      query: {
        enabled: !!expertId,
        refetchInterval: (query) => {
          if (query.state.status === "error") return false;
          if (Date.now() - mountedAtRef.current > POLL_CAP_MS) return false;
          const raw = query.state.data;
          const sessions = raw && raw.status === 200 ? raw.data.sessions : [];
          const hasLive = sessions.some(
            (s) => s.is_processing || s.chat_status === "running",
          );
          // Once a live session is found, SubSessionLive takes over polling
          // it directly by id — this list poll has nothing left to watch for.
          return hasLive ? false : POLL_MS;
        },
        select: (res) => (res.status === 200 ? res.data.sessions : []),
      },
    },
  );
  const live = (data ?? []).find(
    (s) => s.is_processing || s.chat_status === "running",
  );
  return live?.id ?? null;
}

/** A sub-session tool output freezes the status it had when the tool
 *  returned — a "running" card would say running forever after the work
 *  lands. While the frozen status is running/queued, read the truth off the
 *  polled session and flip to completed once it goes idle.
 *
 *  Owns the poll through `useLiveSubSession` rather than piggybacking on a
 *  mounted live view: a minimal delegate card renders no live view, so this
 *  is the only thing left that can flip running → completed. On a full card
 *  it is the same query key, so the two share one poll. */
export function useSubSessionEffectiveStatus(
  subSessionId: string | null,
  status: string | null,
) {
  const stale = ["running", "queued"].includes(status?.toLowerCase() ?? "");
  const { session, isError, isPaused } = useLiveSubSession(
    subSessionId ?? "",
    stale && !!subSessionId,
  );
  if (!stale) return status;
  // The frozen status is only trustworthy while the poll can refute it. A
  // minimal card has no "Live updates paused" notice to fall back on, so a
  // dead poll has to show up in the pill or it reads as fact.
  if (isError || isPaused) return "unknown";
  if (!session) return status;
  return isSessionLive(session) ? status : "completed";
}

function isSessionLive(session: SessionDetailResponse): boolean {
  if (session.active_stream) return true;
  const status = session.chat_status?.toLowerCase();
  return status === "running" || status === "queued";
}

function parseArguments(rawArguments: unknown): unknown {
  if (typeof rawArguments !== "string") return rawArguments ?? {};
  try {
    return JSON.parse(rawArguments);
  } catch {
    return {};
  }
}
