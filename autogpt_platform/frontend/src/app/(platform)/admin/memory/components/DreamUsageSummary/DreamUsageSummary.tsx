"use client";

import type { DreamPassUsage } from "@/app/api/__generated__/models/dreamPassUsage";
import { Text } from "@/components/atoms/Text/Text";
import { Badge } from "@/components/atoms/Badge/Badge";
import { CoinsIcon } from "@phosphor-icons/react";
import { formatCost, formatTokens } from "./helpers";

interface Props {
  usage: DreamPassUsage | null | undefined;
}

export function DreamUsageSummary({ usage }: Props) {
  if (!usage) {
    return (
      <div
        className="rounded-md border border-dashed bg-white p-4 text-center text-sm text-gray-500"
        data-testid="dream-usage-empty"
      >
        No token usage was recorded for this dream pass.
      </div>
    );
  }

  const phases = usage.phases ?? [];
  const discount = usage.discount_applied ?? 0;

  return (
    <div
      className="rounded-md border bg-white p-3"
      data-testid="dream-usage-summary"
    >
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <Text variant="small-medium" className="uppercase text-gray-500">
          Token usage
        </Text>
        <div className="inline-flex items-center gap-2">
          <CoinsIcon size={14} className="text-gray-500" />
          <Text variant="small-medium" as="span">
            total: {formatCost(usage.total_cost_usd)}
          </Text>
          {discount > 0 ? (
            <Badge variant="success" size="small">
              discount {(discount * 100).toFixed(0)}%
            </Badge>
          ) : null}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="text-gray-500">
            <tr>
              <th className="px-2 py-1 font-medium">Phase</th>
              <th className="px-2 py-1 font-medium">Model</th>
              <th className="px-2 py-1 text-right font-medium">In</th>
              <th className="px-2 py-1 text-right font-medium">Out</th>
              <th className="px-2 py-1 text-right font-medium">Cache R</th>
              <th className="px-2 py-1 text-right font-medium">Cache W</th>
              <th className="px-2 py-1 text-right font-medium">Cost</th>
            </tr>
          </thead>
          <tbody>
            {phases.length === 0 ? (
              <tr>
                <td
                  colSpan={7}
                  className="px-2 py-2 text-center italic text-gray-500"
                >
                  No per-phase data recorded.
                </td>
              </tr>
            ) : (
              phases.map((p, i) => (
                <tr
                  key={`${p.phase}-${i}`}
                  className="border-t border-gray-100"
                >
                  <td className="px-2 py-1 font-medium text-gray-800">
                    {p.phase}
                  </td>
                  <td className="px-2 py-1 font-mono text-[10px] text-gray-600">
                    {p.model}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {formatTokens(p.input_tokens)}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {formatTokens(p.output_tokens)}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {formatTokens(p.cache_read_tokens)}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {formatTokens(p.cache_creation_tokens)}
                  </td>
                  <td className="px-2 py-1 text-right tabular-nums">
                    {formatCost(p.cost_usd)}
                  </td>
                </tr>
              ))
            )}
            <tr className="border-t-2 border-gray-200 bg-gray-50 font-medium">
              <td className="px-2 py-1" colSpan={2}>
                Totals
              </td>
              <td className="px-2 py-1 text-right tabular-nums">
                {formatTokens(usage.total_input_tokens)}
              </td>
              <td className="px-2 py-1 text-right tabular-nums">
                {formatTokens(usage.total_output_tokens)}
              </td>
              <td className="px-2 py-1 text-right tabular-nums">
                {formatTokens(usage.total_cache_read_tokens)}
              </td>
              <td className="px-2 py-1 text-right tabular-nums">
                {formatTokens(usage.total_cache_creation_tokens)}
              </td>
              <td className="px-2 py-1 text-right tabular-nums">
                {formatCost(usage.total_cost_usd)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
