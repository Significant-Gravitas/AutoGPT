"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { DashboardSpeed02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import {
  AUTOCOMPACT_TOKENS,
  BASE_CONTEXT_ESTIMATE,
  MODEL_CONTEXT_WINDOW,
  breakdownTotal,
  displayContext,
  formatTokenCount,
  turnInputTokens,
  useTokenDevtoolStore,
  type ContextBreakdown,
  type TokenTurn,
} from "../../tokenDevtool";

interface Props {
  sessionId: string;
}

/** Dev-only context readout in the composer tray. The bar tracks the
 *  estimated live context against the model window, with a marker at the
 *  backend's autocompact trigger — cross it and the next turn is expected
 *  to summarize. The popover splits the estimate by source and lists
 *  per-turn usage; ⟲ marks a turn where the stream carried a
 *  `context_compaction` tool call. */
export function TokenDevtoolBadge({ sessionId }: Props) {
  const turns = useTokenDevtoolStore((s) => s.turnsBySession[sessionId]);
  const breakdown = useTokenDevtoolStore(
    (s) => s.breakdownBySession[sessionId],
  );
  const [open, setOpen] = useState(false);
  const context = displayContext(
    turns,
    breakdown ? breakdownTotal(breakdown) : undefined,
  );

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          aria-label="Token devtool"
          className="ml-auto flex items-center gap-1.5 rounded-full px-2 py-0.5 font-mono text-xs text-zinc-500 transition-colors hover:bg-zinc-200/60 hover:text-zinc-700"
        >
          <Icon icon={DashboardSpeed02Icon} size={14} />
          {context === null ? "ctx —" : `ctx ~${formatTokenCount(context)}`}
          {context !== null && <MiniBar context={context} />}
        </button>
      </PopoverTrigger>

      <PopoverContent
        side="top"
        align="end"
        sideOffset={8}
        className="w-80 rounded-2xl border-zinc-200/70 p-3 shadow-[0_16px_40px_-24px_rgba(0,0,0,0.25)]"
      >
        <div className="flex items-baseline justify-between pb-2">
          <span className="text-sm font-medium text-zinc-800">
            Context window
          </span>
          <span className="font-mono text-xs text-zinc-500">
            {context === null ? "—" : `~${formatTokenCount(context)}`} /{" "}
            {formatTokenCount(MODEL_CONTEXT_WINDOW)}
          </span>
        </div>
        <ContextBar context={context ?? 0} />
        <p className="pb-2.5 pt-1 text-right font-mono text-xs text-amber-500">
          summarizes ~{formatTokenCount(AUTOCOMPACT_TOKENS)}
        </p>

        {breakdown && <BreakdownSection breakdown={breakdown} />}

        {!turns?.length ? (
          <p className="text-sm text-zinc-500">
            Live per-turn data starts with your next message.
          </p>
        ) : (
          <div className="flex max-h-40 flex-col gap-1 overflow-y-auto">
            {turns.map((turn, i) => (
              <TurnRow key={turn.at + i} index={i} turn={turn} />
            ))}
          </div>
        )}
      </PopoverContent>
    </Popover>
  );
}

function pct(context: number): number {
  return Math.min(100, (context / MODEL_CONTEXT_WINDOW) * 100);
}

function MiniBar({ context }: { context: number }) {
  return (
    <span className="relative h-1 w-10 overflow-hidden rounded-full bg-zinc-200">
      <span
        className={
          "absolute inset-y-0 left-0 rounded-full " +
          (context >= AUTOCOMPACT_TOKENS ? "bg-amber-400" : "bg-zinc-500")
        }
        style={{ width: `${pct(context)}%` }}
      />
    </span>
  );
}

function ContextBar({ context }: { context: number }) {
  const threshold = (AUTOCOMPACT_TOKENS / MODEL_CONTEXT_WINDOW) * 100;
  return (
    <div className="relative h-2 w-full rounded-full bg-zinc-100">
      <div
        className={
          "absolute inset-y-0 left-0 rounded-full transition-all " +
          (context >= AUTOCOMPACT_TOKENS ? "bg-amber-400" : "bg-zinc-800")
        }
        style={{ width: `${pct(context)}%` }}
      />
      <div
        className="absolute inset-y-0 w-px bg-amber-400"
        style={{ left: `${threshold}%` }}
        title={`Backend triggers summarization around ${formatTokenCount(AUTOCOMPACT_TOKENS)} tokens`}
      />
    </div>
  );
}

const BREAKDOWN_COLORS = {
  system: "bg-zinc-400",
  user: "bg-sky-400",
  assistant: "bg-violet-400",
  tools: "bg-emerald-400",
};

function BreakdownSection({ breakdown }: { breakdown: ContextBreakdown }) {
  const rows = [
    {
      label: "system + tools + skills",
      tokens: BASE_CONTEXT_ESTIMATE,
      color: BREAKDOWN_COLORS.system,
      note: "fixed est.",
    },
    {
      label: "your messages",
      tokens: breakdown.userTokens,
      color: BREAKDOWN_COLORS.user,
    },
    {
      label: "assistant replies",
      tokens: breakdown.assistantTokens,
      color: BREAKDOWN_COLORS.assistant,
    },
    {
      label: "tool calls + results",
      tokens: breakdown.toolTokens,
      color: BREAKDOWN_COLORS.tools,
    },
  ];
  const total = breakdownTotal(breakdown);

  return (
    <div className="pb-2.5">
      <div className="flex h-1.5 w-full gap-px overflow-hidden rounded-full">
        {rows.map((row) => (
          <div
            key={row.label}
            className={row.color}
            style={{ width: `${(row.tokens / total) * 100}%` }}
          />
        ))}
      </div>
      <div className="flex flex-col gap-0.5 pt-1.5">
        {rows.map((row) => (
          <div
            key={row.label}
            className="flex items-baseline gap-1.5 font-mono text-xs"
          >
            <span className={`size-1.5 shrink-0 rounded-full ${row.color}`} />
            <span className="text-zinc-600">{row.label}</span>
            {row.note && (
              <span className="text-[10px] text-zinc-400">{row.note}</span>
            )}
            <span className="ml-auto text-zinc-500">
              ~{formatTokenCount(row.tokens)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TurnRow({ index, turn }: { index: number; turn: TokenTurn }) {
  return (
    <div className="flex items-baseline gap-2 font-mono text-xs">
      <span className="w-6 shrink-0 text-zinc-500">#{index + 1}</span>
      {turn.compacted && (
        <span
          title="Transcript summarized this turn"
          className="text-amber-500"
        >
          ⟲
        </span>
      )}
      <span className="text-zinc-800">
        in {formatTokenCount(turnInputTokens(turn))}
      </span>
      <span className="text-zinc-500">
        out {formatTokenCount(turn.completionTokens)}
      </span>
      <span className="ml-auto text-zinc-400">
        w {formatTokenCount(turn.cacheCreationTokens)}
      </span>
    </div>
  );
}
