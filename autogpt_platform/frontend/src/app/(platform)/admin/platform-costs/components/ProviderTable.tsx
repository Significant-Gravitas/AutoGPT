import type { ProviderCostSummary } from "@/app/api/__generated__/models/providerCostSummary";
import {
  DEFAULT_COST_PER_RUN,
  estimateCostForRow,
  formatMicrodollars,
  trackingValue,
} from "../helpers";
import { trackingBadge } from "./TrackingBadge";

interface Props {
  data: ProviderCostSummary[];
  costPerRunOverrides: Record<string, number>;
  onCostOverride: (provider: string, val: number) => void;
}

function ProviderTable({ data, costPerRunOverrides, onCostOverride }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-4 py-3">Provider</th>
            <th className="px-4 py-3">Type</th>
            <th className="px-4 py-3 text-right">Usage</th>
            <th className="px-4 py-3 text-right">Requests</th>
            <th className="px-4 py-3 text-right">Known Cost</th>
            <th className="px-4 py-3 text-right">Est. Cost</th>
            <th className="px-4 py-3 text-right">$/run</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const est = estimateCostForRow(row, costPerRunOverrides);
            const tt = row.tracking_type || "per_run";
            const showCostInput = tt === "per_run";
            return (
              <tr
                key={`${row.provider}-${row.tracking_type}`}
                className="border-b hover:bg-muted"
              >
                <td className="px-4 py-3 font-medium">{row.provider}</td>
                <td className="px-4 py-3">
                  {trackingBadge(row.tracking_type)}
                </td>
                <td className="px-4 py-3 text-right">{trackingValue(row)}</td>
                <td className="px-4 py-3 text-right">
                  {row.request_count.toLocaleString()}
                </td>
                <td className="px-4 py-3 text-right">
                  {row.total_cost_microdollars > 0
                    ? formatMicrodollars(row.total_cost_microdollars)
                    : "-"}
                </td>
                <td className="px-4 py-3 text-right">
                  {est !== null ? (
                    formatMicrodollars(est)
                  ) : (
                    <span className="text-muted-foreground">-</span>
                  )}
                </td>
                <td className="px-4 py-2 text-right">
                  {showCostInput ? (
                    <input
                      type="number"
                      step="0.001"
                      min="0"
                      className="w-20 rounded border px-2 py-1 text-right text-xs"
                      placeholder={String(
                        DEFAULT_COST_PER_RUN[row.provider] ?? "0",
                      )}
                      defaultValue={
                        costPerRunOverrides[row.provider] ??
                        DEFAULT_COST_PER_RUN[row.provider] ??
                        ""
                      }
                      onChange={(e) => {
                        const val = parseFloat(e.target.value);
                        if (!isNaN(val)) onCostOverride(row.provider, val);
                      }}
                    />
                  ) : (
                    <span className="text-xs text-muted-foreground">auto</span>
                  )}
                </td>
              </tr>
            );
          })}
          {data.length === 0 && (
            <tr>
              <td
                colSpan={7}
                className="px-4 py-8 text-center text-muted-foreground"
              >
                No cost data yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export { ProviderTable };
