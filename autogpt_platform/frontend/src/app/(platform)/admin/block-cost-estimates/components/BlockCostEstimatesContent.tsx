"use client";

import { DownloadSimpleIcon } from "@phosphor-icons/react";
import { Button } from "@/components/atoms/Button/Button";
import { buildEstimatesJson, downloadJson } from "../helpers";
import { useBlockCostEstimates } from "./useBlockCostEstimates";

export function BlockCostEstimatesContent() {
  const {
    start,
    end,
    minSamples,
    data,
    loading,
    setStart,
    setEnd,
    setMinSamples,
    fetchEstimates,
  } = useBlockCostEstimates();

  function handleDownload() {
    if (!data) return;
    const generatedAtIso =
      data.generated_at instanceof Date
        ? data.generated_at.toISOString()
        : String(data.generated_at);
    const json = buildEstimatesJson(
      data.estimates,
      generatedAtIso,
      data.window_days,
    );
    downloadJson(json, `block_preflight_estimates_${start}_${end}.json`);
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-end gap-3 rounded border p-4">
        <div className="flex flex-col gap-1">
          <label htmlFor="bce-start" className="text-sm">
            Start date (UTC)
          </label>
          <input
            id="bce-start"
            type="date"
            className="rounded border px-3 py-1.5 text-sm"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="bce-end" className="text-sm">
            End date (UTC)
          </label>
          <input
            id="bce-end"
            type="date"
            className="rounded border px-3 py-1.5 text-sm"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="bce-min-samples" className="text-sm">
            Min samples
          </label>
          <input
            id="bce-min-samples"
            type="number"
            min={1}
            className="w-28 rounded border px-3 py-1.5 text-sm"
            value={minSamples}
            onChange={(e) =>
              setMinSamples(Math.max(1, Number(e.target.value) || 1))
            }
          />
        </div>
        <Button
          variant="primary"
          size="small"
          onClick={fetchEstimates}
          loading={loading}
        >
          Aggregate
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={handleDownload}
          disabled={!data || data.total_rows === 0}
          leftIcon={<DownloadSimpleIcon weight="bold" />}
        >
          Download JSON
        </Button>
      </div>

      {data ? (
        <div className="flex flex-col gap-2">
          <p className="text-sm text-muted-foreground">
            {data.total_rows} blocks · window {data.window_days}d (cap{" "}
            {data.max_window_days}d) · min samples {data.min_samples} ·
            generated{" "}
            {data.generated_at instanceof Date
              ? data.generated_at.toISOString()
              : String(data.generated_at)}
          </p>
          <div className="overflow-x-auto rounded border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="p-2 text-left">Block ID</th>
                  <th className="p-2 text-left">Block name</th>
                  <th className="p-2 text-left">Cost type</th>
                  <th className="p-2 text-right">Samples</th>
                  <th className="p-2 text-right">Mean (credits)</th>
                  <th className="p-2 text-right">P50</th>
                  <th className="p-2 text-right">P95</th>
                </tr>
              </thead>
              <tbody>
                {data.estimates.map((r) => (
                  <tr key={r.block_id} className="border-t">
                    <td className="p-2 font-mono text-xs">{r.block_id}</td>
                    <td className="p-2">{r.block_name}</td>
                    <td className="p-2">{r.cost_type}</td>
                    <td className="p-2 text-right">{r.samples}</td>
                    <td className="p-2 text-right font-semibold">
                      {r.mean_credits}
                    </td>
                    <td className="p-2 text-right">{r.p50_credits}</td>
                    <td className="p-2 text-right">{r.p95_credits}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <p className="text-sm text-muted-foreground">
          Pick a window and click Aggregate to compute per-block average
          credits-per-execution.
        </p>
      )}
    </div>
  );
}
