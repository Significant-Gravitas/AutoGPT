import {
  BASE_CONTEXT_ESTIMATE,
  breakdownTotal,
  formatTokenCount,
  type ContextBreakdown,
} from "../../../tokenDevtool/tokenMath";
import { BREAKDOWN_COLORS } from "../helpers";

interface Props {
  breakdown: ContextBreakdown;
}

export function BreakdownSection({ breakdown }: Props) {
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
