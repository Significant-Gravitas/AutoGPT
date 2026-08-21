"use client";

import {
  useGetV2GetSession,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import type { ToolUIPart } from "ai";
import { useRef } from "react";
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
  const mountedAtRef = useRef(Date.now());
  const { data } = useGetV2GetSession(subSessionId, undefined, {
    query: {
      enabled: active && !!subSessionId,
      refetchInterval: (query) => {
        if (Date.now() - mountedAtRef.current > POLL_CAP_MS) return false;
        const raw = query.state.data;
        const session = raw && raw.status === 200 ? raw.data : null;
        return !session || isSessionLive(session) ? POLL_MS : false;
      },
      select: (res) => (res.status === 200 ? res.data : null),
    },
  });
  if (!active || !data) return null;

  const isLive = isSessionLive(data);
  const allMessages = Array.isArray(data.messages) ? data.messages : [];
  // A re-delegation reuses the same sub-session, so only the CURRENT turn
  // (everything after the last user message) belongs to this card — the
  // full history would replay the previous delegation's final answer here.
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
  const recent = steps.slice(-MAX_STEPS);
  if (recent.length === 0 && !latestText) return null;

  const rows = recent.map((step, i) =>
    toMiniRow(step, i, isLive && i === recent.length - 1),
  );

  return (
    <div className="mt-2 border-t border-zinc-100 pl-1 pt-2.5">
      {rows.map((row, i) => {
        const isLastRow = i === rows.length - 1;
        return (
          <div key={row.key} className="flex items-stretch gap-2.5">
            <div className="flex w-6 flex-col items-center">
              <div className="flex size-6 shrink-0 items-center justify-center rounded-full bg-zinc-100">
                <RowIcon row={row} />
              </div>
              {!isLastRow && <div className="w-px flex-1 bg-zinc-200" />}
            </div>
            <div
              className={cn("min-w-0 flex-1 pt-[2px]", !isLastRow && "pb-2.5")}
            >
              <SwapText
                text={row.text}
                shimmer={row.state === "running"}
                className="max-w-full text-sm leading-5 text-zinc-600"
              />
            </div>
          </div>
        );
      })}
      {latestText && (
        <p className="mt-1.5 line-clamp-3 text-sm leading-relaxed text-zinc-600">
          {latestText}
        </p>
      )}
    </div>
  );
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
export function SubSessionPendingCard({ input }: { input: unknown }) {
  const { expertsById } = useExpertMap();
  const args = asObject(input) ?? {};
  const inputExpertId = str(args, "expert_id");
  const prompt = str(args, "prompt");
  const inputSessionId = str(args, "sub_session_id") ?? str(args, "session_id");
  const discoveredId = useDelegatedSessionId(
    inputSessionId ? null : inputExpertId,
  );
  const liveSessionId = inputSessionId ?? discoveredId;
  // Cache-shared with SubSessionLive's query below — no extra request.
  const { data: liveSession } = useGetV2GetSession(
    liveSessionId ?? "",
    undefined,
    {
      query: {
        enabled: !!liveSessionId,
        select: (res) => (res.status === 200 ? res.data : null),
      },
    },
  );
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
        <StatusPill status="running" className="text-sm" />
      </div>
      {prompt && (
        <p className="mt-1.5 line-clamp-2 pl-9 text-sm text-zinc-500">
          {prompt}
        </p>
      )}
      {liveSessionId && <SubSessionLive subSessionId={liveSessionId} active />}
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
    { expert_id: expertId ?? undefined, limit: 5 },
    {
      query: {
        enabled: !!expertId,
        refetchInterval: () =>
          Date.now() - mountedAtRef.current > POLL_CAP_MS ? false : POLL_MS,
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
 *  polled session (cache-shared with the live view) and flip to completed
 *  once it goes idle. */
export function useSubSessionEffectiveStatus(
  subSessionId: string | null,
  status: string | null,
) {
  const stale = ["running", "queued"].includes(status?.toLowerCase() ?? "");
  const { data } = useGetV2GetSession(subSessionId ?? "", undefined, {
    query: {
      enabled: stale && !!subSessionId,
      select: (res) => (res.status === 200 ? res.data : null),
    },
  });
  if (!stale || !data) return status;
  return isSessionLive(data) ? status : "completed";
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
