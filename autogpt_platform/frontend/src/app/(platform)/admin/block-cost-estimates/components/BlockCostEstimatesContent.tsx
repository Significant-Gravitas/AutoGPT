"use client";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { buildEstimatesJson, downloadJson } from "../helpers";
import { useBlockCostEstimates } from "./useBlockCostEstimates";
import { Download04Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

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
          <Input
            id="bce-start"
            label="Start date (UTC)"
            labelVariant="body"
            size="small"
            wrapperClassName="mb-0"
            type="date"
            value={start}
            onChange={(e) => setStart(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="bce-end"
            label="End date (UTC)"
            labelVariant="body"
            size="small"
            wrapperClassName="mb-0"
            type="date"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="bce-min-samples"
            label="Min samples"
            labelVariant="body"
            size="small"
            wrapperClassName="mb-0"
            type="number"
            min={1}
            className="w-28"
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
          leftIcon={<Icon icon={Download04Icon} />}
        >
          Download JSON
        </Button>
      </div>

      {data ? (
        <div className="flex flex-col gap-2">
          <Text variant="body" tone="muted">
            {data.total_rows} blocks · window {data.window_days}d (cap{" "}
            {data.max_window_days}d) · min samples {data.min_samples} ·
            generated{" "}
            {data.generated_at instanceof Date
              ? data.generated_at.toISOString()
              : String(data.generated_at)}
          </Text>
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
        <Text variant="body" tone="muted">
          Pick a window and click Aggregate to compute per-block average
          credits-per-execution.
        </Text>
      )}
    </div>
  );
}
