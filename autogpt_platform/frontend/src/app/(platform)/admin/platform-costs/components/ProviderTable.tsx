import type { ProviderCostSummary } from "@/app/api/__generated__/models/providerCostSummary";
import {
  defaultRateFor,
  estimateCostForRow,
  formatMicrodollars,
  rateKey,
  rateUnitLabel,
  trackingValue,
} from "../helpers";
import { TrackingBadge } from "./TrackingBadge";

interface Props {
  data: ProviderCostSummary[];
  rateOverrides: Record<string, number>;
  onRateOverride: (key: string, val: number | null) => void;
}

function ProviderTable({ data, rateOverrides, onRateOverride }: Props) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b text-xs uppercase text-muted-foreground">
          <tr>
            <th scope="col" className="px-4 py-3">
              Provider
            </th>
            <th scope="col" className="px-4 py-3">
              Type
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Usage
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Requests
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Known Cost
            </th>
            <th scope="col" className="px-4 py-3 text-right">
              Est. Cost
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-right"
              title="Per-session only"
            >
              Rate <span className="text-[10px] font-normal">(unsaved)</span>
            </th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => {
            const est = estimateCostForRow(row, rateOverrides);
            const tt = row.tracking_type || "per_run";
            // For cost_usd rows the provider reports USD directly so rate
            // input doesn't apply; otherwise show an editable input.
            const showRateInput = tt !== "cost_usd";
            const key = rateKey(row.provider, tt);
            const fallback = defaultRateFor(row.provider, tt);
            const currentRate = rateOverrides[key] ?? fallback;
            return (
              <tr key={key} className="border-b hover:bg-muted">
                <td className="px-4 py-3 font-medium">{row.provider}</td>
                <td className="px-4 py-3">
                  <TrackingBadge trackingType={row.tracking_type} />
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
                  {showRateInput ? (
                    <div className="flex items-center justify-end gap-1">
                      <input
                        type="number"
                        step="0.0001"
                        min="0"
                        aria-label={`Rate for ${row.provider} (${tt})`}
                        className="w-24 rounded border px-2 py-1 text-right text-xs"
                        placeholder={fallback !== null ? String(fallback) : "0"}
                        value={currentRate ?? ""}
                        onChange={(e) => {
                          const val = parseFloat(e.target.value);
                          if (!isNaN(val)) onRateOverride(key, val);
                          else if (e.target.value === "")
                            onRateOverride(key, null);
                        }}
                      />
                      <span
                        className="text-[10px] text-muted-foreground"
                        title={rateUnitLabel(tt)}
                      >
                        {rateUnitLabel(tt)}
                      </span>
                    </div>
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
