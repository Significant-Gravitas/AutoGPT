"use client";

/*
 * Evals – a prompt evaluation dashboard for teams shipping LLM features, in a single file.
 * A header with the prompt, its version and model, four key figures, the pass rate over the last
 * twelve versions next to the reviewable changes in the system prompt, then the test cases across
 * the full width. "Run eval" walks through
 * the cases one by one with a thin progress bar and ends with a toast.
 * Sits in the affectUI App Shell; uses the KPI Chart, Inline Diff, Segmented Control and Button from
 * the library, icons from the Hugeicons free set.
 *
 * AutoGPT admin: installed with `shadcn add @affectui/evals` into this folder. The layout, classes, motion and
 * styles below are the block's own. What changed, so it shows the saved eval runs: the sample data became
 * props (EvalsContent.tsx feeds them), a version is a saved run, the prompt diff card shows what changed since
 * the previous run and how each category did, "Run eval" opens the commands that make a run locally (a run
 * spends model money, so the page never starts one), the cases table gets a category menu and a search, and a
 * case opens in the page's own side panel (EvalsContent.tsx, styled like the chat's work area). The page sits in the
 * admin area's own sidebar, so the App Shell's frame is used without its sidebar (ShellFrame, app-shell.tsx). The run
 * simulation and its progress bar are gone.
 */

import { useState, type CSSProperties } from "react";
import {
  ArrowDown01Icon,
  ArrowLeft01Icon,
  ArrowRight01Icon,
  ArrowRight02Icon,
  CommandLineIcon,
  Copy01Icon,
  CpuIcon,
  FilterIcon,
  GitCommitIcon,
  TestTube01Icon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/app/(platform)/admin/evals/affect/ui/button-clean";
import { KpiChart } from "@/app/(platform)/admin/evals/affect/ui/kpi-chart-clean";
import { SegmentedControl } from "@/app/(platform)/admin/evals/affect/ui/segmented-control-clean";
import {
  Dropdown,
  EmptyState,
  Icon,
  Loading,
  MenuButton,
  Popover,
  SearchField,
  ShellFrame,
  Skeleton,
  useToast,
} from "@/app/(platform)/admin/evals/affect/app-shell";

/** One saved run of the same setup: its label, pass rate in % and when it ran */
export type EvalVersion = {
  id: string;
  label: string;
  rate: number;
  edited: string;
};

type Verdict = "pass" | "partial" | "fail";
export type EvalStatus = Verdict | "skipped";
export type EvalCase = {
  id: string;
  category: string;
  input: string;
  expected: string;
  output: string;
  /** 0–100; null when the case was skipped */
  score: number | null;
  status: EvalStatus;
  /** "3/4", "2 failed" or "–" */
  checks: string;
  /** Everything the search box looks through, lowercase */
  search: string;
};
export type EvalStat = {
  label: string;
  value: string;
  unit?: string;
  delta: string;
  good: boolean;
  note: string;
};
export type EvalRunOption = { id: string; label: string; description: string };
export type EvalCommandLine = { label: string; command: string; cost: string };
export type EvalCaseChange = {
  id: string;
  title: string;
  before: Verdict;
  after: Verdict;
};
export type EvalCategoryLine = {
  name: string;
  label: string;
  rate: number;
  delta: number | null;
  cases: number;
  failed: number;
};
export type EvalChanges = {
  /** "Since Baseline native run 4" – null on the first run of its setup */
  since: string | null;
  categories: EvalCategoryLine[];
  /** False when one of the two runs kept only its totals */
  compared: boolean;
  worse: EvalCaseChange[];
  better: EvalCaseChange[];
};

const verdictLabel: Record<EvalStatus, string> = {
  pass: "Pass",
  partial: "Partial",
  fail: "Fail",
  skipped: "Skipped",
};
const noneLabel: Record<Verdict, string> = {
  pass: "No passing cases",
  partial: "No partial cases",
  fail: "No failing cases",
};

/** Test cases per page in the table */
const PAGE = 50;
const SHOWN_CHANGES = 6;
const ALL = "all";
const count = new Intl.NumberFormat("en-US");

type Filter = "all" | Verdict;

export type EvalsScreenProps = {
  className?: string;
  /** The open run's name in the header */
  title?: string;
  /** Model the run used */
  model?: string;
  /** Every run of the open run's setup with its pass rate, oldest first */
  versions?: EvalVersion[];
  /** The open run */
  runId?: string;
  /** Every saved run, for the picker */
  runs?: EvalRunOption[];
  onPickRun?: (id: string) => void;
  stats?: EvalStat[];
  cases?: EvalCase[];
  changes?: EvalChanges | null;
  /** A line over the cases, e.g. that a checked-in run kept only its failures */
  note?: string | null;
  /** "Ran 9 Nov, 08:41" */
  meta?: string;
  commands?: EvalCommandLine[];
  /** The open case – it shows in the page's side panel */
  selected?: string | null;
  onSelect?: (id: string) => void;
  /** While your data loads: gray key figures, chart, card and test case rows in the shape of the real ones */
  loading?: boolean;
  /** No run saved yet: the header stays, the cases show an empty state with the commands */
  empty?: boolean;
  /** The run could not be read */
  error?: { title: string; text: string; onRetry: () => void } | null;
  /** The run kept no cases: where to find one that did */
  emptyCases?: {
    text: string;
    action?: { label: string; onClick: () => void };
  } | null;
};

export function EvalsScreen({
  title = "Evals",
  model = "",
  versions: versionList = [],
  runId,
  runs = [],
  onPickRun,
  stats = [],
  cases = [],
  changes = null,
  note,
  meta = "",
  commands = [],
  selected = null,
  onSelect,
  loading,
  empty: emptyProp,
  error,
  emptyCases,
  className,
}: EvalsScreenProps) {
  // Loading wins over empty
  const empty = !!emptyProp && !loading;
  const versions = versionList.map((v) => v.label);
  const history = versionList.map((v) =>
    Number.isFinite(v.rate) ? v.rate : 0,
  );
  const edited = versionList.map((v) => v.edited);
  const vIndex = Math.max(
    0,
    versionList.findIndex((v) => v.id === runId),
  );
  const [filter, setFilterState] = useState<Filter>("all");
  const [category, setCategoryState] = useState(ALL);
  const [search, setSearchState] = useState("");
  const [page, setPage] = useState(0);
  const setFilter = (next: Filter) => {
    setFilterState(next);
    setPage(0);
  };
  const setCategory = (next: string) => {
    setCategoryState(next);
    setPage(0);
  };
  const setSearch = (next: string) => {
    setSearchState(next);
    setPage(0);
  };
  const [commandsOpen, setCommandsOpen] = useState(false);
  const [toast, showToast] = useToast();
  const open = (id: string) => onSelect?.(id);
  const copy = (command: EvalCommandLine) => {
    navigator.clipboard
      .writeText(command.command)
      .then(() => showToast(`Copied · ${command.label}`))
      .catch(() => showToast("Copy failed – select the command instead"));
  };

  const rateDelta =
    (history[vIndex] ?? 0) - (history[vIndex - 1] ?? history[vIndex] ?? 0);
  const categories = Array.from(new Set(cases.map((c) => c.category))).sort();
  const query = search.trim().toLowerCase();
  const inScope = cases.filter(
    (c) =>
      (category === ALL || c.category === category) &&
      (!query || c.search.includes(query)),
  );
  const counts = { pass: 0, partial: 0, fail: 0 };
  inScope.forEach((c) => {
    if (c.status !== "skipped") counts[c.status]++;
  });
  const shown = inScope
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => filter === "all" || c.status === filter);
  // 50 cases per page, so thousands of cases stay quick
  const pages = Math.max(1, Math.ceil(shown.length / PAGE));
  const pageNow = Math.min(page, pages - 1);
  const rows = shown.slice(pageNow * PAGE, pageNow * PAGE + PAGE);
  const current = runs.find((r) => r.id === runId);

  return (
    <ShellFrame className={["ui-ev-host", className].filter(Boolean).join(" ")}>
      <style href="ui-evals" precedence="default">
        {css}
      </style>
      <div className="ui-ev">
        <header className="ui-ev__head">
          <div className="ui-ev__title">
            {/* A quiet path like the other screens: section, then the run that is open */}
            <h1 className="ui-ev__path" title={title}>
              <span>Evals</span>
              <i>/</i>
              <b>{title}</b>
            </h1>
            {loading ? (
              <span className="ui-ev__chip ui-ev__chip--btn" data-skel="">
                <Skeleton width={56} height={8} />
                <Icon icon={ArrowDown01Icon} size={13} />
              </span>
            ) : runs.length ? (
              <Dropdown
                title="Saved runs"
                items={runs.map((r) => ({
                  label: r.label,
                  icon: GitCommitIcon,
                  description: r.description,
                  checked: r.id === runId,
                  onSelect: () => onPickRun?.(r.id),
                }))}
                trigger={({ open: isOpen, toggle }) => (
                  <button
                    type="button"
                    className="ui-ev__chip ui-ev__chip--btn"
                    aria-label={`Choose a run: ${current?.label ?? title}`}
                    aria-expanded={isOpen}
                    aria-haspopup="menu"
                    onClick={toggle}
                  >
                    {versions[vIndex]
                      ? `#${vIndex + 1} of ${versions.length}`
                      : "Runs"}
                    {vIndex === versions.length - 1 && versions.length > 0 && (
                      <span className="ui-ev__current">latest</span>
                    )}
                    <Icon icon={ArrowDown01Icon} size={13} />
                  </button>
                )}
              />
            ) : null}
            {model && (
              <span className="ui-ev__chip" title={model}>
                <Icon icon={CpuIcon} size={14} />
                <span>{model}</span>
              </span>
            )}
          </div>
          <div className="ui-ev__actions">
            <span className="ui-ev__meta" aria-live="polite">
              {loading ? "" : meta}
            </span>
            <span className="ui-as-anchor">
              <Button
                size="sm"
                onClick={() => setCommandsOpen((o) => !o)}
                disabled={!commands.length}
                icon={<Icon icon={CommandLineIcon} size={14} />}
              >
                Run locally
              </Button>
              <Popover
                open={commandsOpen}
                onClose={() => setCommandsOpen(false)}
                align="end"
                className="ui-as-menu ui-ev__cmds"
              >
                <Commands commands={commands} onCopy={copy} />
              </Popover>
            </span>
          </div>
        </header>

        <div className="ui-ev__content" data-empty={empty || undefined}>
          {empty || error ? null : loading ? (
            <Loading label="Loading key figures" className="ui-ev__stats">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="ui-ev__card ui-ev__stat" data-skel="">
                  <span className="ui-ev__cap">
                    <Skeleton width={72} height={8} />
                  </span>
                  {/* An unseen figure and delta keep the card's real height (they sit on one baseline); the bars lie over them */}
                  <strong className="ui-ev__skel-ghost">
                    <span aria-hidden="true">
                      0<small>%</small>
                    </span>
                    <Skeleton width={[52, 44, 40, 56][i]} height={14} />
                  </strong>
                  <span className="ui-ev__delta ui-ev__skel-ghost">
                    <span aria-hidden="true">
                      0 <span>vs</span>
                    </span>
                    <Skeleton width={[64, 72, 68, 76][i]} height={8} />
                  </span>
                </div>
              ))}
            </Loading>
          ) : (
            <div className="ui-ev__stats">
              {stats.map((s, i) => (
                <div
                  key={s.label}
                  className="ui-ev__card ui-ev__stat ui-as-rise"
                  style={{ "--i": i } as CSSProperties}
                >
                  <span className="ui-ev__cap">{s.label}</span>
                  <strong>
                    {s.value}
                    {s.unit && <small>{s.unit}</small>}
                  </strong>
                  <span
                    className="ui-ev__delta"
                    data-good={s.good || undefined}
                  >
                    {s.delta} {s.note && <span>{s.note}</span>}
                  </span>
                </div>
              ))}
            </div>
          )}

          {empty || error ? null : loading ? (
            <Loading
              label="Loading pass rate"
              className="ui-ev__card ui-ev__chart-skel"
            >
              <div className="ui-ev__chart-skel-body" data-skel="">
                <span className="ui-ev__cap">Pass rate by run</span>
                <span className="ui-ev__skel-num ui-ev__skel-value">
                  <Skeleton width={52} height={14} />
                </span>
                <div className="ui-ev__chart-skel-grid">
                  <span className="ui-ev__chart-skel-y">
                    <Skeleton width={22} height={6} />
                    <Skeleton width={22} height={6} />
                    <Skeleton width={22} height={6} />
                  </span>
                  <span className="ui-ev__chart-skel-plot">
                    {[38, 44, 52, 49, 58, 66, 63, 72, 76, 70, 82, 90].map(
                      (h, i) => (
                        <Skeleton
                          key={i}
                          width="auto"
                          height={`${h}%`}
                          radius={4}
                        />
                      ),
                    )}
                  </span>
                  <span className="ui-ev__chart-skel-x">
                    <Skeleton width={18} height={6} />
                    <Skeleton width={18} height={6} />
                    <Skeleton width={24} height={6} />
                  </span>
                </div>
              </div>
              <div className="ui-ev__chart-skel-foot" data-skel="">
                <Skeleton width={220} height={8} />
              </div>
            </Loading>
          ) : (
            <KpiChart
              className="ui-ev__chart"
              title="Pass rate by run"
              value={`${history[vIndex] ?? 0}%`}
              delta={Math.round(rateDelta * 10) / 10}
              format={(n) => `${Math.round(n)}%`}
              labels={versions.map((v, i) => shortLabel(v, i))}
              tooltipLabels={versions.map((v, i) => `${v} · ${edited[i]}`)}
              series={[
                { label: "Pass rate", color: "#4fb0fe", values: history },
              ]}
              defaultIndex={vIndex === versions.length - 1 ? undefined : vIndex}
              footer={`${count.format(versions.length)} ${versions.length === 1 ? "run" : "runs"} of this setup · ${count.format(cases.length)} ${cases.length === 1 ? "case" : "cases"} saved · ${model}`}
            />
          )}

          {!empty && !error && loading && (
            <Loading
              label="Loading changes"
              className="ui-ev__card ui-ev__diff-skel"
            >
              <span className="ui-ev__diff-skel-label" data-skel="">
                <Skeleton width={236} height={8} />
              </span>
              <div className="ui-ev__diff-skel-body" data-skel="">
                {["96%", "88%", "100%", "56%"].map((w, i) => (
                  <span key={i} className="ui-ev__diff-skel-line">
                    <Skeleton width={w} height={10} />
                  </span>
                ))}
              </div>
            </Loading>
          )}

          {!empty && !error && !loading && changes && (
            <Changes
              changes={changes}
              onOpen={open}
              known={new Set(cases.map((c) => c.id))}
            />
          )}

          <section
            className="ui-ev__cases"
            data-note={(note && !loading && !error) || undefined}
          >
            <div className="ui-ev__cases-head">
              <h2>
                Test cases{" "}
                {!loading && !empty && !error && (
                  <span>{count.format(cases.length)}</span>
                )}
              </h2>
              <div className="ui-ev__tools">
                {!loading && cases.length > 0 && (
                  <>
                    <SearchField
                      placeholder="Search cases and reasons"
                      width={210}
                      value={search}
                      onChange={setSearch}
                    />
                    <MenuButton
                      value={category}
                      icon={FilterIcon}
                      onChange={setCategory}
                      options={[
                        { value: ALL, label: "All categories" },
                        ...categories.map((c) => ({
                          value: c,
                          label: categoryLabel(c),
                        })),
                      ]}
                    />
                  </>
                )}
                <SegmentedControl<Filter>
                  size="sm"
                  value={filter}
                  onChange={setFilter}
                  options={[
                    { value: "all", label: "All" },
                    {
                      value: "pass",
                      label: "Pass",
                      count: loading ? undefined : counts.pass,
                    },
                    {
                      value: "partial",
                      label: "Partial",
                      count: loading ? undefined : counts.partial,
                    },
                    {
                      value: "fail",
                      label: "Fail",
                      count: loading ? undefined : counts.fail,
                    },
                  ]}
                />
              </div>
            </div>
            {note && !loading && !error && (
              <p className="ui-ev__note">{note}</p>
            )}
            {loading ? (
              <Loading label="Loading test cases" className="ui-ev__scroll">
                <table className="ui-ev__table">
                  <thead>
                    <tr>
                      <th>Request</th>
                      <th>Expected</th>
                      <th>Otto did</th>
                      <th>Score</th>
                      <th>Checks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      [72, 64, 80],
                      [58, 76, 68],
                      [80, 52, 74],
                      [64, 70, 60],
                      [76, 58, 82],
                      [54, 66, 70],
                      [70, 74, 56],
                      [62, 60, 78],
                    ].map(([a, b, c], i) => (
                      <tr key={i} data-skel="">
                        <td>
                          <span className="ui-ev__skel-cell">
                            <Skeleton width={36} height={8} />
                            <Skeleton width={`${a}%`} height={10} />
                          </span>
                        </td>
                        <td>
                          <span className="ui-ev__skel-cell">
                            <Skeleton width={`${b}%`} height={10} />
                          </span>
                        </td>
                        <td>
                          <span className="ui-ev__skel-cell">
                            <Skeleton width={`${c}%`} height={10} />
                          </span>
                        </td>
                        <td>
                          <span className="ui-ev__skel-cell">
                            <Skeleton width={68} height={20} radius={6} />
                          </span>
                        </td>
                        <td>
                          <span className="ui-ev__skel-cell ui-ev__skel-cell--end">
                            <Skeleton width={28} height={8} />
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Loading>
            ) : error ? (
              <EmptyState
                icon={TestTube01Icon}
                title={error.title}
                text={error.text}
                action={
                  <Button variant="secondary" size="sm" onClick={error.onRetry}>
                    Try again
                  </Button>
                }
              />
            ) : empty ? (
              <EmptyState
                icon={TestTube01Icon}
                title="No eval runs saved yet"
                text="Runs show up here once the eval writes its results. Make one locally, then reload."
                action={
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<Icon icon={CommandLineIcon} size={15} />}
                    onClick={() => setCommandsOpen(true)}
                  >
                    Show the commands
                  </Button>
                }
              />
            ) : !cases.length ? (
              <EmptyState
                icon={TestTube01Icon}
                title="No cases saved for this run"
                text={emptyCases?.text ?? "Only this run's totals were kept."}
                action={
                  emptyCases?.action ? (
                    <Button
                      variant="secondary"
                      size="sm"
                      onClick={emptyCases.action.onClick}
                    >
                      {emptyCases.action.label}
                    </Button>
                  ) : undefined
                }
              />
            ) : !shown.length ? (
              <EmptyState
                bare
                icon={FilterIcon}
                title={
                  filter !== "all" && !query && category === ALL
                    ? noneLabel[filter]
                    : "No cases match"
                }
                text="Every case in this run landed outside these filters."
              />
            ) : (
              <div className="ui-ev__scroll">
                <table className="ui-ev__table">
                  <thead>
                    <tr>
                      <th>Request</th>
                      <th>Expected</th>
                      <th>Otto did</th>
                      <th>Score</th>
                      <th>Checks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ c, i }) => (
                      <tr
                        key={c.id}
                        data-i={i}
                        aria-selected={c.id === selected}
                        aria-label={`${c.id}: ${verdictLabel[c.status]}`}
                        data-state={c.status}
                        tabIndex={0}
                        onClick={() => open(c.id)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter" || e.key === " ") {
                            e.preventDefault();
                            open(c.id);
                          }
                        }}
                      >
                        <td>
                          <span className="ui-ev__in">
                            <span className="ui-ev__id">{c.id}</span>
                            <span className="ui-ev__trunc" title={c.input}>
                              {c.input}
                            </span>
                          </span>
                        </td>
                        <td>
                          <span
                            className="ui-ev__trunc ui-ev__soft"
                            title={c.expected}
                          >
                            {c.expected}
                          </span>
                        </td>
                        <td>
                          <span
                            className="ui-ev__trunc ui-ev__soft"
                            title={c.output}
                          >
                            {c.output}
                          </span>
                        </td>
                        <td>
                          <Pill status={c.status} score={c.score} />
                        </td>
                        <td className="ui-ev__num">{c.checks}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                {pages > 1 && (
                  <div className="ui-ev__pager">
                    <span>
                      {count.format(pageNow * PAGE + 1)}–
                      {count.format(pageNow * PAGE + rows.length)} of{" "}
                      {count.format(shown.length)}
                    </span>
                    <button
                      type="button"
                      aria-label="Previous page"
                      disabled={pageNow === 0}
                      onClick={() => setPage(pageNow - 1)}
                    >
                      <Icon icon={ArrowLeft01Icon} size={14} />
                    </button>
                    <button
                      type="button"
                      aria-label="Next page"
                      disabled={pageNow >= pages - 1}
                      onClick={() => setPage(pageNow + 1)}
                    >
                      <Icon icon={ArrowRight01Icon} size={14} />
                    </button>
                  </div>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
      {toast}
    </ShellFrame>
  );
}

/** In the place of the prompt's diff: what moved since the run before, category by category, then case by case */
function Changes({
  changes,
  onOpen,
  known,
}: {
  changes: EvalChanges;
  onOpen: (id: string) => void;
  known: Set<string>;
}) {
  return (
    <section
      className="ui-ev__card ui-ev__changes"
      aria-label="Changes since the previous run"
    >
      <span className="ui-ev__cap">
        {changes.since ?? "By category · the first run of this setup"}
      </span>
      <ul className="ui-ev__cats">
        {changes.categories.map((c) => (
          <li key={c.name}>
            <span className="ui-ev__trunc">{c.label}</span>
            <span className="ui-ev__bar" aria-hidden="true">
              <i
                style={{ width: `${c.rate}%` }}
                data-v={
                  c.rate >= 80 ? "pass" : c.rate >= 50 ? "partial" : "fail"
                }
              />
            </span>
            <span className="ui-ev__num">{c.rate}%</span>
            <span
              className="ui-ev__delta"
              data-good={c.delta !== null && c.delta >= 0 ? true : undefined}
            >
              {c.delta === null || !changes.since
                ? ""
                : `${c.delta > 0 ? "+" : c.delta < 0 ? "−" : "±"}${Math.abs(c.delta)}`}
            </span>
          </li>
        ))}
      </ul>
      {changes.since && !changes.compared && (
        <p className="ui-ev__note">
          One of the two runs kept only its totals, so cases can&apos;t be
          compared one by one.
        </p>
      )}
      {changes.since && changes.compared && (
        <>
          <CaseChanges
            title="Got worse"
            list={changes.worse}
            onOpen={onOpen}
            known={known}
          />
          <CaseChanges
            title="Got better"
            list={changes.better}
            onOpen={onOpen}
            known={known}
          />
        </>
      )}
    </section>
  );
}

function CaseChanges({
  title,
  list,
  onOpen,
  known,
}: {
  title: string;
  list: EvalCaseChange[];
  onOpen: (id: string) => void;
  known: Set<string>;
}) {
  return (
    <div className="ui-ev__moves">
      <h3>
        {title} <span>{count.format(list.length)}</span>
      </h3>
      {list.slice(0, SHOWN_CHANGES).map((c) => (
        <button
          key={c.id}
          type="button"
          className="ui-ev__move"
          title={c.title}
          disabled={!known.has(c.id)}
          onClick={() => onOpen(c.id)}
        >
          <span className="ui-ev__id">{c.id}</span>
          <Pill status={c.before} />
          <Icon icon={ArrowRight02Icon} size={12} />
          <Pill status={c.after} />
        </button>
      ))}
      {list.length > SHOWN_CHANGES && (
        <span className="ui-ev__more">
          and {count.format(list.length - SHOWN_CHANGES)} more
        </span>
      )}
    </div>
  );
}

function Commands({
  commands,
  onCopy,
}: {
  commands: EvalCommandLine[];
  onCopy: (command: EvalCommandLine) => void;
}) {
  return (
    <div className="ui-ev__cmd-list">
      <p className="ui-ev__note">
        Runs happen on your machine, from autogpt_platform/backend. A case that
        calls a model costs money, so there is no run button here.
      </p>
      {commands.map((c) => (
        <div key={c.command} className="ui-ev__cmd">
          <span className="ui-ev__cmd-head">
            <b>{c.label}</b>
            <span>{c.cost}</span>
          </span>
          <span className="ui-ev__cmd-line">
            <code>{c.command}</code>
            <button
              type="button"
              aria-label={`Copy: ${c.label}`}
              onClick={() => onCopy(c)}
            >
              <Icon icon={Copy01Icon} size={14} />
            </button>
          </span>
        </div>
      ))}
    </div>
  );
}

function Pill({
  status,
  score,
}: {
  status: EvalStatus;
  score?: number | null;
}) {
  return (
    <span className="ui-ev__pill" data-v={status} key={status + (score ?? "")}>
      <i aria-hidden="true" />
      {verdictLabel[status]}
      {score !== undefined && score !== null && (
        <span>{Math.round(score)}</span>
      )}
    </span>
  );
}

function shortLabel(label: string, i: number) {
  const run = /run (\d+)$/.exec(label);
  return run ? `#${run[1]}` : `${i + 1}`;
}

function categoryLabel(name: string) {
  const words = name.replaceAll("_", " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/* ---------- Styles ---------- */

const css = /* css */ `
/* Evals: header with prompt, version and model, then cards on the panel */
.ui-ev {
  --ui-ev-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  --ev-pass: var(--aui-success, #1e8a4f);
  --ev-pass-soft: color-mix(in srgb, var(--aui-green, #5fc88a) 10%, transparent);
  --ev-partial: var(--aui-warning, #b25e00);
  --ev-partial-dot: var(--aui-amber, #e8b000);
  --ev-partial-soft: color-mix(in srgb, var(--aui-amber, #e8b000) 13%, transparent);
  --ev-fail: var(--aui-red, #f04438);
  --ev-fail-soft: color-mix(in srgb, var(--aui-red, #f04438) 9%, transparent);
  --ev-run: var(--aui-blue, #4fb0fe);
  --ev-run-soft: color-mix(in srgb, var(--aui-blue, #0190fe) 10%, transparent);
  display: flex;
  flex-direction: column;
  min-height: 100%;
}
:where(.dark, [data-theme="dark"]) .ui-ev {
  --ui-ev-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
  --ev-pass: var(--aui-green, #5fc88a);
  --ev-pass-soft: color-mix(in srgb, var(--aui-green, #7dcf9f) 12%, transparent);
  --ev-partial: var(--aui-amber, #e6c24d);
  --ev-partial-soft: color-mix(in srgb, var(--aui-amber, #e6c24d) 13%, transparent);
  --ev-fail: var(--aui-red, #ff6b5f);
  --ev-fail-soft: color-mix(in srgb, var(--aui-red, #ff6b5f) 13%, transparent);
  --ev-run: var(--aui-blue, #4fb0fe);
  --ev-run-soft: color-mix(in srgb, var(--aui-blue, #4fb0fe) 15%, transparent);
}

/* The bar stays on top while the page scrolls under it */
.ui-ev__head { position: sticky; top: 0; z-index: 6; display: flex; flex: none; align-items: center; justify-content: space-between; gap: 12px; height: var(--as-head-h, 48px); padding: 0 12px 0 20px; background: var(--panel); }
.ui-ev__title { display: flex; align-items: center; gap: 8px; min-width: 0; }
.ui-ev__path { display: flex; flex: 0 1 auto; align-items: center; gap: 8px; min-width: 0; margin: 0 4px 0 0; font-size: 13px; font-weight: 400; color: var(--muted); white-space: nowrap; }
.ui-ev__path i { font-style: normal; color: var(--faint); }
/* Long names stay on one line: the title and the model give way first, the full text is in their title */
.ui-ev__path b { min-width: 0; overflow: hidden; font-weight: 500; text-overflow: ellipsis; color: var(--ink); }
.ui-ev__chip > span { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
.ui-ev__chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 26px;
  padding: 0 8px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 2px);
  background: var(--chip);
  font-size: 12px;
  color: var(--text);
  white-space: nowrap;
}
.ui-ev__chip:not(.ui-ev__chip--btn) { flex: 0 10 auto; min-width: 96px; }
.ui-ev__chip > svg { flex: none; }
.ui-ev__chip--btn { flex: none; }
.ui-ev__chip:not(.ui-ev__chip--btn) { font-family: var(--aui-font-mono, ui-monospace, "SF Mono", Menlo, monospace); font-size: 12px; }
.ui-ev__chip--btn { font-weight: 500; color: var(--ink); transition: background-color 140ms ease; }
.ui-ev__chip--btn:hover, .ui-ev__chip--btn[aria-expanded="true"] { background: var(--chip-hover); }
.ui-ev__current { font-weight: 400; color: var(--muted); }
.ui-ev__actions { display: flex; flex: none; align-items: center; gap: 12px; }
.ui-ev__meta { font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; white-space: nowrap; }

/* Thin progress line under the header while a run is going */
.ui-ev__progress { position: absolute; left: 16px; right: 16px; bottom: 0; height: 2px; border-radius: 1px; overflow: hidden; background: var(--line); opacity: 0; transition: opacity 300ms ease; }
.ui-ev__progress[data-on] { opacity: 1; }
.ui-ev__progress i { display: block; height: 100%; border-radius: 1px; background: var(--ev-run); transition: width 200ms var(--ease); }

/* Two columns for the chart and the diff; the test cases span both below */
.ui-ev__content {
  display: grid;
  flex: 1;
  grid-template-columns: minmax(0, 1fr) 400px;
  grid-template-rows: auto auto minmax(380px, 1fr);
  gap: 12px;
  padding: 12px 16px 16px;
}
.ui-ev__card { border-radius: calc(var(--aui-radius-lg, 10px) + 2px); background: var(--card); box-shadow: var(--card-shadow); }
.ui-ev__cap { font-size: 12.5px; line-height: 18px; color: var(--muted); }

.ui-ev__stats { grid-column: 1 / -1; display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.ui-ev__stat { display: grid; grid-template-columns: 1fr auto; align-items: baseline; row-gap: 10px; padding: 14px 16px; }
.ui-ev__stat .ui-ev__cap { grid-column: 1 / -1; }
.ui-ev__stat strong { font-size: 20px; line-height: 28px; font-weight: 500; letter-spacing: -0.02em; color: var(--ink); font-variant-numeric: tabular-nums; }
.ui-ev__stat small { margin-left: 1px; font-size: 13px; font-weight: 400; letter-spacing: 0; color: var(--faint); }
.ui-ev__delta { font-size: 12px; font-weight: 500; color: var(--ev-fail); font-variant-numeric: tabular-nums; }
.ui-ev__delta[data-good] { color: var(--ev-pass); }
.ui-ev__delta span { font-weight: 400; color: var(--faint); }

/* Library chart flat, with the same edge as the cards; its plot grows with the row */
.ui-ev .ui-ev__chart.ui-kpic {
  --ui-kpic-shadow: var(--card-shadow);
  display: flex;
  flex-direction: column;
  width: 100%;
  max-width: none;
  border: 0;
  border-radius: calc(var(--aui-radius-lg, 10px) + 2px);
}
/* Without a prompt change to show, the chart takes the full width */
.ui-ev .ui-ev__chart--wide.ui-kpic { grid-column: 1 / -1; }
.ui-ev .ui-ev__chart .ui-kpic__chart { flex: 1; grid-template-rows: minmax(140px, 1fr) 20px; }
/* The chart's title and figure in the same sizes as the stat cards – like the Analytics screen */
.ui-ev .ui-ev__chart .ui-kpic__title { font-size: 12.5px; font-weight: 400; line-height: 18px; color: var(--muted); }
.ui-ev .ui-ev__chart .ui-kpic__value { margin-top: 10px; font-size: 20px; line-height: 28px; }
.ui-ev .ui-ev__chart .ui-kpic__footer { font-size: 12px; }

/* Inline Diff next to the chart: one white card as tall as the chart – its label inside on top like the chart's title,
   room under the text, the buttons at the bottom without a line above them */
.ui-ev .ui-ev__diff.ui-dfc { display: flex; flex-direction: column; align-self: stretch; padding: 14px 0 0; border-radius: calc(var(--aui-radius-lg, 10px) + 2px); background: var(--card); box-shadow: var(--card-shadow); }
.ui-ev .ui-ev__diff .ui-dfc__label { margin: 0 16px 4px; font-size: 12.5px; line-height: 18px; color: var(--muted); }
.ui-ev .ui-ev__diff .ui-dfc__card { display: flex; flex: 1; flex-direction: column; border-radius: calc(var(--aui-radius-lg, 10px) + 2px); background: none; box-shadow: none; }
.ui-ev .ui-ev__diff .ui-dfc__body { flex: 1; padding: 6px 16px 20px; }
.ui-ev .ui-ev__diff .ui-dfc__foot { border-top: 0; }
.ui-ev .ui-ev__diff .ui-dfc__text { font-size: 13px; line-height: 1.6; }

/* Test cases: the table scrolls inside its card, so the row always fits the screen */
/* Full width, with the same room above it as a section in the other screens (38px from the content above) */
/* Not clipped, so the table card's edge shows all round */
.ui-ev__cases { position: relative; display: flex; flex-direction: column; grid-column: 1 / -1; min-height: 0; margin-top: 26px; }
.ui-ev__cases-head { display: flex; flex: none; align-items: center; justify-content: space-between; gap: 12px; height: 32px; margin-bottom: 10px; padding-left: 2px; }
.ui-ev__cases-head h2 { display: flex; align-items: center; gap: 6px; margin: 0; font-size: 14px; line-height: 20px; font-weight: 500; color: var(--ink); }
.ui-ev__cases-head h2 span { font-weight: 400; color: var(--muted); }
/* One white card like the invoices in Settings / Billing: a quiet head row, then rows split by hairlines – it scrolls inside */
.ui-ev__scroll { position: absolute; inset: 42px 0 0; overflow: auto; overscroll-behavior: contain; border-radius: calc(var(--aui-radius-lg, 10px) + 2px); background: var(--card); box-shadow: var(--card-shadow); }
/* No cases at all: the empty state fills the table's place; a filter with no match: centered on the panel */
.ui-ev__cases > .ui-as-empty { position: absolute; inset: 42px 0 0; justify-content: center; margin-top: 0; }
/* A new prompt: only the test cases, filling the page */
.ui-ev__content[data-empty] { grid-template-rows: minmax(380px, 1fr); }
.ui-ev__content[data-empty] .ui-ev__cases { margin-top: 0; }
/* Pages of 50 cases under the table */
.ui-ev__pager { display: flex; align-items: center; justify-content: flex-end; gap: 4px; padding: 8px 10px; box-shadow: inset 0 0.5px 0 var(--line); font-size: 12.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
.ui-ev__pager span { margin-right: 6px; }
.ui-ev__pager button { display: grid; place-items: center; width: 28px; height: 28px; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); color: var(--muted); }
.ui-ev__pager button:hover:not(:disabled) { background: var(--hover); color: var(--ink); }
.ui-ev__pager button:disabled { opacity: 0.4; cursor: default; }
.ui-ev__table { width: 100%; border-collapse: separate; border-spacing: 0; table-layout: fixed; font-size: 13px; }
.ui-ev__table th {
  position: sticky;
  z-index: 1;
  top: 0;
  height: 40px;
  padding: 0 12px;
  background: var(--card);
  box-shadow: inset 0 -0.5px 0 var(--line);
  font-weight: 400;
  font-size: 12px;
  text-align: left;
  color: var(--muted);
}
.ui-ev__table th:nth-child(1) { width: 30%; padding-left: 18px; }
.ui-ev__table th:nth-child(2) { width: 22%; }
.ui-ev__table th:nth-child(4) { width: 112px; }
.ui-ev__table th:nth-child(5) { width: 80px; padding-right: 18px; text-align: right; }
.ui-ev__table td { height: 48px; padding: 0 12px; background: var(--card); color: var(--ink); transition: background-color 140ms ease; }
.ui-ev__table tbody tr + tr td { box-shadow: inset 0 0.5px 0 var(--line); }
.ui-ev__table td:first-child { padding-left: 18px; }
.ui-ev__table td:last-child { padding-right: 18px; }
.ui-ev__in { display: flex; align-items: center; gap: 8px; min-width: 0; }
.ui-ev__table tbody tr { cursor: pointer; }
.ui-ev__table tbody tr:hover td { background: color-mix(in srgb, var(--card), var(--ink) 2.5%); }
.ui-ev__table tbody tr[aria-selected="true"] td { background: color-mix(in srgb, var(--card), var(--ink) 4%); }
.ui-ev__table tbody tr[data-state="running"] td { background: linear-gradient(var(--ev-run-soft), var(--ev-run-soft)), var(--card); }
.ui-ev__table tbody tr:focus-visible { outline: 2px solid var(--ui-ev-focus); outline-offset: -2px; }
.ui-ev__table tbody tr[data-state="queued"] td { color: var(--faint); }
.ui-ev__trunc { display: block; min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.ui-ev__soft { color: var(--muted); }
.ui-ev__id { flex: none; font-family: var(--aui-font-mono, ui-monospace, "SF Mono", Menlo, monospace); font-size: 11.5px; color: var(--faint); }
.ui-ev__num { text-align: right; color: var(--muted) !important; font-variant-numeric: tabular-nums; }

/* Score pill – pass, partial, fail, or the run state */
.ui-ev__pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  height: 20px;
  padding: 0 7px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  font-size: 11.5px;
  font-weight: 500;
  white-space: nowrap;
  animation: ui-ev-pop 320ms var(--ease) both;
}
.ui-ev__pill > i:not(.ui-ev__spin) { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
.ui-ev__pill > span { font-weight: 400; opacity: 0.75; font-variant-numeric: tabular-nums; }
.ui-ev__pill[data-v="pass"] { background: var(--ev-pass-soft); color: var(--ev-pass); }
.ui-ev__pill[data-v="partial"] { background: var(--ev-partial-soft); color: var(--ev-partial); }
.ui-ev__pill[data-v="fail"] { background: var(--ev-fail-soft); color: var(--ev-fail); }
.ui-ev__pill[data-v="queued"] { background: var(--chip); color: var(--faint); font-weight: 400; animation: none; }
.ui-ev__pill[data-v="running"] { background: var(--ev-run-soft); color: var(--ev-run); animation: none; }
.ui-ev__spin { width: 10px; height: 10px; border: 1.5px solid currentColor; border-right-color: transparent; border-radius: 50%; animation: ui-ev-spin 700ms linear infinite; }
@keyframes ui-ev-pop { from { opacity: 0; transform: scale(0.85); } to { opacity: 1; transform: none; } }
@keyframes ui-ev-spin { to { transform: rotate(1turn); } }

/* Selected case: input, expected, last version's output against this one's */
.ui-ev__input { margin: -4px 0 0; font-size: 15px; font-weight: 500; line-height: 1.45; color: var(--ink); }
.ui-ev__field { display: grid; gap: 4px; }
.ui-ev__field p { margin: 0; font-size: 13px; line-height: 1.5; color: var(--text); }
.ui-ev__foot { display: flex; align-items: center; gap: 6px; margin-top: auto; font-size: 12px; color: var(--faint); }

/* Loading: the same cards, chart box, diff card and table, gray shapes in place of the data – nothing moves when it lands */
.ui-ev [data-skel] { pointer-events: none; }
/* Each bar centered in the line of what it stands in for: the 20px figure's 28px line, a table cell's 20px line */
.ui-ev__skel-num { display: flex; align-items: center; height: 28px; }
.ui-ev__skel-ghost { position: relative; }
.ui-ev__skel-ghost > span { visibility: hidden; }
.ui-ev__skel-ghost > .ui-as-skel { position: absolute; top: 50%; left: 0; translate: 0 -50%; }
.ui-ev__delta.ui-ev__skel-ghost > .ui-as-skel { right: 0; left: auto; }
.ui-ev__skel-cell { display: flex; align-items: center; gap: 8px; min-width: 0; height: 20px; }
.ui-ev__skel-cell--end { justify-content: flex-end; }
/* The KPI Chart's box: title, figure, plot with y and x labels, footer – the same paddings and rows as the library chart here */
.ui-ev__chart-skel { display: flex; flex-direction: column; min-width: 0; }
.ui-ev__chart-skel--wide { grid-column: 1 / -1; }
.ui-ev__chart-skel-body { display: flex; flex: 1; flex-direction: column; padding: 14px 16px 12px; }
.ui-ev__chart-skel-body > .ui-ev__cap { display: block; }
.ui-ev__skel-value { margin-top: 10px; }
.ui-ev__chart-skel-grid { display: grid; flex: 1; grid-template-columns: auto 1fr; grid-template-rows: minmax(140px, 1fr) 20px; column-gap: 10px; row-gap: 8px; margin-top: 20px; }
.ui-ev__chart-skel-y { display: flex; flex-direction: column; justify-content: space-between; }
.ui-ev__chart-skel-plot { display: flex; align-items: flex-end; gap: 6px; min-width: 0; }
.ui-ev__chart-skel-plot > .ui-as-skel { flex: 1 1 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
.ui-ev__chart-skel-x { display: flex; grid-column: 2; align-items: center; justify-content: space-between; }
.ui-ev__chart-skel-foot { display: flex; align-items: center; height: 36.5px; padding: 0 16px; border-top: 0.5px solid var(--line); }
/* The Inline Diff's card: label, the prompt's 20.8px lines, the footer with its buttons */
.ui-ev__diff-skel { display: flex; flex-direction: column; align-self: stretch; min-width: 0; padding-top: 14px; }
.ui-ev__diff-skel-label { display: flex; align-items: center; height: 18px; margin: 0 16px 4px; }
.ui-ev__diff-skel-body { flex: 1; padding: 6px 16px 20px; }
.ui-ev__diff-skel-line { display: flex; align-items: center; height: 20.8px; }
.ui-ev__diff-skel-foot { display: flex; align-items: center; justify-content: space-between; min-height: 44.5px; padding: 8px 8px 8px 16px; border-top: 0.5px solid var(--line); }
.ui-ev__diff-skel-btns { display: flex; gap: 8px; }

/* Touch: the version chip, Run and the page buttons grow to a comfortable 32px */
@media (pointer: coarse) {
  .ui-ev__chip--btn { height: 32px; }
  .ui-ev__pager button { width: 32px; height: 32px; }
}

/* Room for the sidebar icon on narrow screens, where the sidebar is a drawer */
.ui-ev__head { transition: padding-left 420ms var(--ease); }

/* Narrow panels: one column; table and detail get their own height */
@container ui-as-main (max-width: 1000px) {
  .ui-ev__content { grid-template-columns: minmax(0, 1fr); grid-template-rows: none; }
  .ui-ev__cases { height: 440px; }
}
@container ui-as-main (max-width: 820px) {
  .ui-ev__stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .ui-ev__table { min-width: 720px; }
}
@container ui-as-main (max-width: 560px) {
  .ui-ev__stats { gap: 8px; }
  .ui-ev__stat { padding: 12px; }
  .ui-ev__stat strong { font-size: 19px; }
  .ui-ev__cases-head { flex-wrap: wrap; }
  .ui-ev__scroll, .ui-ev__cases > .ui-as-empty { top: 84px; }
}
@container ui-as (max-width: 900px) {
  .ui-ev__head { padding-left: 52px; }
  .ui-ev__meta { display: none; }
}
@container ui-as (max-width: 600px) {
  /* Title next to the sidebar icon, version, model and Run on a row below */
  /* The 20px title in the middle of a head row – on the sidebar icon's centerline */
  .ui-ev__head { flex-wrap: wrap; row-gap: 10px; height: auto; padding: calc((var(--as-head-h, 48px) - 20px) / 2) 8px 12px 52px; }
  .ui-ev__path { line-height: 20px; }
  /* Version, model and Run share the second row, so Run doesn't sit alone on a third one */
  .ui-ev__title { display: contents; }
  .ui-ev__title h1 { width: 100%; margin-right: 0; }
  .ui-ev__head { column-gap: 8px; }
  .ui-ev__chip:not(.ui-ev__chip--btn) { flex: 1 1 0; max-width: max-content; }
  .ui-ev__actions { margin-left: auto; }
  .ui-ev__content { gap: 8px; padding: 8px 8px 8px; }
}

/* ---------- AutoGPT admin additions, in the block's own style ---------- */
/* The cases head carries the search and the category menu next to the result filter */
.ui-ev__tools { display: flex; align-items: center; gap: 8px; min-width: 0; }
/* A quiet line under the head – e.g. that a checked-in run kept only its failures; the table moves down for it */
.ui-ev__note { margin: 0; font-size: 12.5px; line-height: 18px; color: var(--muted); }
.ui-ev__cases > .ui-ev__note { position: absolute; top: 42px; left: 2px; right: 2px; }
.ui-ev__cases[data-note] > .ui-ev__scroll, .ui-ev__cases[data-note] > .ui-as-empty { top: 84px; }
/* A skipped case: gray like a queued one */
.ui-ev__pill[data-v="skipped"] { background: var(--chip); color: var(--faint); font-weight: 400; }
.ui-ev__table tbody tr[data-state="fail"] td:first-child { box-shadow: inset 2px 0 0 var(--ev-fail); }
.ui-ev__table tbody tr + tr[data-state="fail"] td:first-child { box-shadow: inset 2px 0 0 var(--ev-fail), inset 0 0.5px 0 var(--line); }

/* In the place of the prompt's diff: one card as tall as the chart – categories, then what got worse and better */
.ui-ev__changes { display: flex; flex-direction: column; gap: 12px; min-width: 0; padding: 14px 16px 16px; }
.ui-ev__cats { display: grid; gap: 7px; margin: 0; padding: 0; list-style: none; font-size: 12.5px; color: var(--text); }
.ui-ev__cats li { display: grid; grid-template-columns: minmax(0, 1fr) 72px 40px 36px; align-items: center; gap: 10px; }
.ui-ev__cats .ui-ev__num { color: var(--ink) !important; }
.ui-ev__cats .ui-ev__delta { text-align: right; }
.ui-ev__bar { display: block; height: 4px; border-radius: 2px; background: var(--chip); overflow: hidden; }
.ui-ev__bar i { display: block; height: 100%; border-radius: 2px; transform-origin: left center; animation: ui-ev-grow 520ms var(--ease) both; }
.ui-ev__bar i[data-v="pass"] { background: var(--ev-pass); }
.ui-ev__bar i[data-v="partial"] { background: var(--ev-partial-dot, var(--ev-partial)); }
.ui-ev__bar i[data-v="fail"] { background: var(--ev-fail); }
@keyframes ui-ev-grow { from { transform: scaleX(0); } to { transform: none; } }
.ui-ev__moves { display: grid; gap: 4px; padding-top: 12px; box-shadow: inset 0 0.5px 0 var(--line); }
.ui-ev__moves h3 { display: flex; gap: 6px; margin: 0 0 2px; font-size: 12.5px; font-weight: 500; color: var(--ink); }
.ui-ev__moves h3 span { font-weight: 400; color: var(--muted); font-variant-numeric: tabular-nums; }
.ui-ev__move { display: flex; align-items: center; gap: 6px; height: 28px; margin: 0 -6px; padding: 0 6px; border-radius: calc(var(--aui-radius-sm, 5px) + 2px); color: var(--faint); text-align: left; transition: background-color 140ms ease; }
.ui-ev__move .ui-ev__id { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
.ui-ev__move:hover:not(:disabled) { background: var(--hover); }
.ui-ev__move:disabled { cursor: default; }
.ui-ev__move:focus-visible { outline: 2px solid var(--ui-ev-focus); outline-offset: -2px; }
.ui-ev__more { font-size: 12px; color: var(--muted); }

/* "Run locally": the commands in a menu card under the button */
.ui-ev__cmds { width: min(560px, 92vw); padding: 12px; }
.ui-ev__cmd-list { display: grid; gap: 12px; }
.ui-ev__cmd { display: grid; gap: 6px; }
.ui-ev__cmd-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; font-size: 12.5px; color: var(--muted); }
.ui-ev__cmd-head b { font-weight: 500; color: var(--ink); }
.ui-ev__cmd-line { display: flex; align-items: flex-start; gap: 8px; padding: 8px 8px 8px 10px; border-radius: calc(var(--aui-radius-sm, 5px) + 3px); background: var(--chip); }
.ui-ev__cmd-line code { flex: 1; min-width: 0; font-family: var(--aui-font-mono, ui-monospace, "SF Mono", Menlo, monospace); font-size: 11.5px; line-height: 1.5; color: var(--text); word-break: break-all; }
.ui-ev__cmd-line button { display: grid; flex: none; place-items: center; width: 24px; height: 24px; border-radius: calc(var(--aui-radius-sm, 5px) + 1px); color: var(--muted); }
.ui-ev__cmd-line button:hover { background: var(--chip-hover); color: var(--ink); }

/* Hosted in the admin area without the shell's sidebar: the frame scrolls, and the key figures keep their own height –
   the cases take the rest of the screen and scroll inside their card */
.ui-ev-host { container: ui-as ui-as-main / inline-size; position: relative; height: 100%; overflow: auto; overscroll-behavior: contain; background: var(--panel); }
.ui-ev-host .ui-ev__content { align-content: start; grid-template-rows: auto auto auto; }
.ui-ev-host .ui-ev__cases { height: max(440px, calc(100dvh - 140px)); }
.ui-ev-host .ui-ev__content[data-empty] .ui-ev__cases { height: max(380px, calc(100dvh - 90px)); }
.ui-ev-host .ui-ev__cases:has(> .ui-as-empty:not([data-bare])) { height: 360px; }
/* The chart's plot grows with its row, as the block means it to (this library version wraps it in a body) */
.ui-ev-host .ui-ev__chart .ui-kpic__body { display: flex; flex: 1; flex-direction: column; min-height: 0; }
/* No shell sidebar to make room for on narrow screens */
@container ui-as (max-width: 900px) { .ui-ev-host .ui-ev__head { padding-left: 20px; } .ui-ev-host .ui-ev__meta { display: inline; } }
@container ui-as (max-width: 600px) { .ui-ev-host .ui-ev__head { padding-left: 8px; } }
@media (prefers-reduced-motion: reduce) {
  .ui-ev__pill, .ui-ev__bar i { animation: none; }
}
`;
