import type { BlockCostEstimateRow } from "@/app/api/__generated__/models/blockCostEstimateRow";

export function dateInputToUtcIso(input: string): string | null {
  if (!input) return null;
  return new Date(`${input}T00:00:00Z`).toISOString();
}

export function dateInputToUtcIsoEnd(input: string): string | null {
  if (!input) return null;
  return new Date(`${input}T23:59:59.999Z`).toISOString();
}

export function defaultStartDate(): string {
  // 6 days back so the inclusive [start, end] window covers exactly 7
  // calendar days when end is today.
  const d = new Date();
  d.setUTCDate(d.getUTCDate() - 6);
  return d.toISOString().slice(0, 10);
}

export function defaultEndDate(): string {
  return new Date().toISOString().slice(0, 10);
}

export function buildEstimatesJson(
  rows: BlockCostEstimateRow[],
  generatedAt: string,
  windowDays: number,
): string {
  const payload = {
    version: 1,
    generated_at: generatedAt,
    source_window_days: windowDays,
    estimates: Object.fromEntries(
      rows.map((r) => [
        r.block_id,
        {
          block_name: r.block_name,
          cost_type: r.cost_type,
          samples: r.samples,
          mean_credits: r.mean_credits,
        },
      ]),
    ),
  };
  return JSON.stringify(payload, null, 2) + "\n";
}

export function downloadJson(json: string, filename: string): void {
  const blob = new Blob([json], { type: "application/json;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
