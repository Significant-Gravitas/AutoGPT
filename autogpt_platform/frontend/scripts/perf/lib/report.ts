import fs from "fs";
import path from "path";

export interface RunMetrics {
  commits: number;
  totalRenders: number;
  rendersPerCommit: number;
  maxRendersPerCommit: number;
  scriptMs: number;
  taskMs: number;
  layoutMs: number;
  recalcStyleMs: number;
  longTaskCount: number;
  longTaskMs: number;
  heapUsedMB: number;
}

export interface RankedEntry {
  key: string;
  count: number;
}

export interface RunResult {
  scenario: string;
  label: string;
  run: number;
  startedAt: string;
  gitSha: string | null;
  baseURL: string;
  windowMs: number;
  metrics: RunMetrics;
  components: RankedEntry[];
  causes: RankedEntry[];
  pageErrors: number;
  cpuProfile?: string;
}

interface MetricRow {
  key: keyof RunMetrics;
  label: string;
  digits: number;
}

export const METRIC_ROWS: MetricRow[] = [
  { key: "commits", label: "commits", digits: 0 },
  { key: "totalRenders", label: "total renders", digits: 0 },
  { key: "rendersPerCommit", label: "renders per commit", digits: 1 },
  { key: "maxRendersPerCommit", label: "max renders in one commit", digits: 0 },
  { key: "scriptMs", label: "script time (ms)", digits: 0 },
  { key: "taskMs", label: "task time (ms)", digits: 0 },
  { key: "layoutMs", label: "layout time (ms)", digits: 0 },
  { key: "recalcStyleMs", label: "style recalc time (ms)", digits: 0 },
  { key: "longTaskCount", label: "long tasks (≥ 50 ms)", digits: 0 },
  { key: "longTaskMs", label: "long task total (ms)", digits: 0 },
  { key: "heapUsedMB", label: "heap used after GC (MB)", digits: 1 },
];

const TOP_N = 10;

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2
    ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function medianRanking(lists: RankedEntry[][]): RankedEntry[] {
  const keys = new Set(lists.flatMap((list) => list.map((entry) => entry.key)));
  const perRun = lists.map(
    (list) => new Map(list.map((entry) => [entry.key, entry.count])),
  );
  return Array.from(keys, (key) => ({
    key,
    count: median(perRun.map((counts) => counts.get(key) ?? 0)),
  }))
    .filter((entry) => entry.count > 0)
    .sort((a, b) => b.count - a.count || a.key.localeCompare(b.key));
}

export function summarize(runs: RunResult[]) {
  const metrics = Object.fromEntries(
    METRIC_ROWS.map(({ key }) => [
      key,
      median(runs.map((run) => run.metrics[key])),
    ]),
  ) as unknown as RunMetrics;
  return {
    metrics,
    components: medianRanking(runs.map((run) => run.components)),
    causes: medianRanking(runs.map((run) => run.causes)),
  };
}

function formatNumber(value: number, digits: number) {
  return value.toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

export function formatDelta(a: number, b: number, digits: number) {
  const diff = b - a;
  const sign = diff > 0 ? "+" : "";
  const absolute = `${sign}${formatNumber(diff, digits)}`;
  if (a === 0) return diff === 0 ? "0" : absolute;
  const percent = (diff / Math.abs(a)) * 100;
  return `${absolute} (${sign}${percent.toFixed(1)}%)`;
}

function table(header: string[], rows: string[][]) {
  const line = (cells: string[]) => `| ${cells.join(" | ")} |`;
  return [line(header), line(header.map(() => "---")), ...rows.map(line)].join(
    "\n",
  );
}

function rankingTable(title: string, entries: RankedEntry[]) {
  const rows = entries
    .slice(0, TOP_N)
    .map((entry) => [`\`${entry.key}\``, formatNumber(entry.count, 0)]);
  return `**${title}**\n\n${table(["name", "renders"], rows)}`;
}

export function formatRun(result: RunResult) {
  const rows = METRIC_ROWS.map(({ key, label, digits }) => [
    label,
    formatNumber(result.metrics[key], digits),
  ]);
  return [
    `### ${result.scenario} · ${result.label} · run ${result.run}`,
    table(["metric", "value"], rows),
    rankingTable("Top components", result.components),
    rankingTable("Top render causes", result.causes),
  ].join("\n\n");
}

interface LabelledRuns {
  label: string;
  runs: RunResult[];
}

interface CompareArgs {
  scenario: string;
  a: LabelledRuns;
  b: LabelledRuns;
}

function sideBySideRanking(
  title: string,
  a: { label: string; entries: RankedEntry[] },
  b: { label: string; entries: RankedEntry[] },
) {
  const countsA = new Map(a.entries.map((entry) => [entry.key, entry.count]));
  const countsB = new Map(b.entries.map((entry) => [entry.key, entry.count]));
  const keys = Array.from(
    new Set([
      ...a.entries.slice(0, TOP_N).map((entry) => entry.key),
      ...b.entries.slice(0, TOP_N).map((entry) => entry.key),
    ]),
  ).sort(
    (x, y) =>
      Math.max(countsB.get(y) ?? 0, countsA.get(y) ?? 0) -
      Math.max(countsB.get(x) ?? 0, countsA.get(x) ?? 0),
  );
  const rows = keys.map((key) => {
    const valueA = countsA.get(key) ?? 0;
    const valueB = countsB.get(key) ?? 0;
    return [
      `\`${key}\``,
      formatNumber(valueA, 0),
      formatNumber(valueB, 0),
      formatDelta(valueA, valueB, 0),
    ];
  });
  return `**${title}** (median renders)\n\n${table(["name", a.label, b.label, "Δ"], rows)}`;
}

function runsTable(sets: LabelledRuns[]) {
  const rows = sets.flatMap(({ label, runs }) =>
    runs.map((run) => [
      label,
      String(run.run),
      run.gitSha?.slice(0, 9) ?? "?",
      formatNumber(run.metrics.commits, 0),
      formatNumber(run.metrics.totalRenders, 0),
      formatNumber(run.metrics.rendersPerCommit, 1),
      formatNumber(run.metrics.scriptMs, 0),
      formatNumber(run.metrics.longTaskCount, 0),
      formatNumber(run.metrics.heapUsedMB, 1),
    ]),
  );
  return table(
    [
      "label",
      "run",
      "commit",
      "commits",
      "renders",
      "renders/commit",
      "script ms",
      "long tasks",
      "heap MB",
    ],
    rows,
  );
}

export function formatComparison({ scenario, a, b }: CompareArgs) {
  const summaryA = summarize(a.runs);
  const summaryB = summarize(b.runs);
  const metricRows = METRIC_ROWS.map(({ key, label, digits }) => [
    label,
    formatNumber(summaryA.metrics[key], digits),
    formatNumber(summaryB.metrics[key], digits),
    formatDelta(summaryA.metrics[key], summaryB.metrics[key], digits),
  ]);
  return [
    `### ${scenario} — median of ${a.runs.length} (${a.label}) vs ${b.runs.length} (${b.label})`,
    table(["metric", a.label, b.label, "Δ"], metricRows),
    sideBySideRanking(
      "Top components",
      { label: a.label, entries: summaryA.components },
      { label: b.label, entries: summaryB.components },
    ),
    sideBySideRanking(
      "Top render causes",
      { label: a.label, entries: summaryA.causes },
      { label: b.label, entries: summaryB.causes },
    ),
    `<details><summary>Raw runs</summary>\n\n${runsTable([a, b])}\n\n</details>`,
  ].join("\n\n");
}

export function spreadPercent(values: number[]) {
  const middle = median(values);
  if (values.length < 2 || middle === 0) return 0;
  return ((Math.max(...values) - Math.min(...values)) / Math.abs(middle)) * 100;
}

export function formatSummary({ label, runs }: LabelledRuns) {
  const rows = METRIC_ROWS.map(({ key, label: name, digits }) => {
    const values = runs.map((run) => run.metrics[key]);
    return [
      name,
      formatNumber(median(values), digits),
      `${formatNumber(Math.min(...values), digits)} – ${formatNumber(Math.max(...values), digits)}`,
      `${spreadPercent(values).toFixed(1)}%`,
    ];
  });
  const scenario = runs[0]?.scenario ?? "?";
  const summary = summarize(runs);
  return [
    `### ${scenario} · ${label} — median of ${runs.length} runs`,
    table(["metric", "median", "range", "spread"], rows),
    rankingTable("Top components (median renders)", summary.components),
    rankingTable("Top render causes (median renders)", summary.causes),
  ].join("\n\n");
}

export function safeLabel(label: string) {
  if (!/^[A-Za-z0-9._-]+$/.test(label)) {
    throw new Error(
      `Label "${label}" may only contain letters, digits, ".", "_" and "-".`,
    );
  }
  return label;
}

export function runFileName(result: RunResult) {
  const stamp = result.startedAt.replace(/[:.]/g, "-");
  return `${result.label}-${stamp}-${result.run}.json`;
}

export function writeRun(dir: string, result: RunResult) {
  fs.mkdirSync(dir, { recursive: true });
  const file = path.join(dir, runFileName(result));
  fs.writeFileSync(file, `${JSON.stringify(result, null, 2)}\n`);
  return file;
}

export function loadRuns(dir: string, label: string): RunResult[] {
  if (!fs.existsSync(dir)) return [];
  return fs
    .readdirSync(dir)
    .filter((name) => name.endsWith(".json"))
    .map(
      (name) =>
        JSON.parse(fs.readFileSync(path.join(dir, name), "utf8")) as RunResult,
    )
    .filter((result) => result.label === label)
    .sort((x, y) => x.startedAt.localeCompare(y.startedAt) || x.run - y.run);
}
