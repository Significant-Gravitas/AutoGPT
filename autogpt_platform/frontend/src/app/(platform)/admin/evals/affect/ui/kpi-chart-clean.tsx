"use client";

import {
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type PointerEvent,
} from "react";

/*
 * KPI Chart (clean style) – a metric card with a multi-line chart and a hover tooltip, in a single self-contained file.
 * One white card with hairline shadows; the footer sits in the card's last row.
 * The styles are injected below, so no extra files are needed.
 */

export type KpiSeries = {
  label: string;
  /** Line color, any CSS color */
  color: string;
  /** One value per label, same length as `labels` */
  values: number[];
};

export type KpiChartProps = {
  title: string;
  /** Headline value, already formatted, e.g. "$96.4k" */
  value: string;
  /** Change against the previous period – positive is green, negative red */
  delta?: number;
  /** Short labels along the x-axis, e.g. "Mar 4" */
  labels: string[];
  /** Longer labels for the tooltip header, e.g. "Tue, Mar 4" – falls back to `labels` */
  tooltipLabels?: string[];
  series: KpiSeries[];
  /** Formats the axis, tooltip and delta values */
  format?: (n: number) => string;
  /** Line under the chart */
  footer?: string;
  /** Point that stays highlighted while nobody hovers – leave empty to hide the tooltip at rest */
  defaultIndex?: number;
  /** Quiet text in the plot when there is nothing to draw */
  emptyText?: string;
  className?: string;
};

const finite = (n: unknown): n is number =>
  typeof n === "number" && Number.isFinite(n);
/** The value of 100% when the formatter writes percentages – 100 for (n) => `${n}%`, 1 for fractions – else 0 */
const percentScale = (format: (n: number) => string) => {
  try {
    const one = String(format(1));
    if (!/%\s*$/.test(one) || !/%\s*$/.test(String(format(50)))) return 0;
    return Math.abs(parseFloat(one.replace(/[^\d.]/g, "")) - 100) < 1e-6
      ? 1
      : 100;
  } catch {
    return 0;
  }
};
/** 0.0004 → "0.0004", 4200 → "4.2k", 12,345,678 → "12M" */
const compact = (n: number) => {
  const a = Math.abs(n);
  if (a >= 1e9) return `${+(n / 1e9).toFixed(a >= 1e10 ? 0 : 1)}B`;
  if (a >= 1e6) return `${+(n / 1e6).toFixed(a >= 1e7 ? 0 : 1)}M`;
  if (a >= 1000) return `${+(n / 1000).toFixed(a >= 10000 ? 0 : 1)}k`;
  if (a > 0 && a < 1) return `${+n.toPrecision(2)}`;
  return `${Math.round(n)}`;
};

/** Rounds the top of the axis up to a value that halves cleanly: 30 → 15 → 0, $8k → $4k → $0 */
function niceMax(max: number) {
  if (!finite(max) || max < 0) max = 0;
  const half = max / 2 || 1;
  const pow = 10 ** Math.floor(Math.log10(half));
  const step =
    [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].find((s) => s * pow >= half) ?? 10;
  return step * pow * 2;
}

export function KpiChart({
  title,
  value,
  delta,
  labels = [],
  tooltipLabels,
  series = [],
  format = compact,
  footer,
  defaultIndex,
  emptyText = "No data yet",
  className,
}: KpiChartProps) {
  const count = labels.length;
  // A point that exists – anything else (out of range, NaN) shows no point
  const clamp = (i: number | null | undefined) =>
    i == null || !finite(i) || count === 0
      ? null
      : Math.min(count - 1, Math.max(0, Math.round(i)));
  const [hover, setHover] = useState<number | null>(null);
  // One value per label – NaN, Infinity and values past the last label are left out
  const vals = useMemo(
    () =>
      series.map((s) =>
        (s.values ?? []).slice(0, count).map((v) => (finite(v) ? v : null)),
      ),
    [series, count],
  );
  const all = vals.flat().filter(finite);
  const percent = percentScale(format);
  // Nothing to draw: no points, no values, or only zeros – except 0% on a percent axis, which is a reading
  const empty = all.length === 0 || (!percent && all.every((v) => v === 0));
  const active = empty ? null : (hover ?? clamp(defaultIndex));
  // The tooltip keeps its last content while it fades out
  const [shown, setShown] = useState(active ?? 0);
  if (active !== null && active !== shown) setShown(active);
  const cur = Math.min(shown, Math.max(0, count - 1));
  // Percent axes get a sensible span: all 0% reads 0–100%, small rates at least 0–10%
  const top = Math.max(0, ...all);
  const max = !percent
    ? niceMax(top)
    : top === 0
      ? percent
      : niceMax(Math.max(top, percent / 10));
  // Negative values get room below zero, mirrored to a round step
  const low = Math.min(0, ...all);
  const min = low < 0 ? -niceMax(-low) : 0;
  const x = (i: number) => (count > 1 ? (i / (count - 1)) * 100 : 50);
  const y = (v: number) => ((max - v) / (max - min)) * 100;
  // A gap where a value is missing, instead of a line through NaN
  const paths = useMemo(
    () =>
      vals.map((vs) => {
        let d = "";
        vs.forEach((v, i) => {
          if (v === null) return;
          d += `${d && vs[i - 1] !== null ? "L" : "M"}${x(i)},${y(v)}`;
        });
        return d;
      }),
    [vals, max, min, count], // eslint-disable-line react-hooks/exhaustive-deps
  );
  // A point with no neighbour draws no line (one version, or a value between gaps) – it gets a dot
  const lone = vals.flatMap((vs, k) =>
    vs.flatMap((v, i) =>
      v !== null && (vs[i - 1] ?? null) === null && (vs[i + 1] ?? null) === null
        ? [{ k, i, v }]
        : [],
    ),
  );

  const indexAt = (e: PointerEvent<HTMLDivElement>) => {
    const box = e.currentTarget.getBoundingClientRect();
    const ratio = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
    return clamp(ratio * (count - 1));
  };

  const onKey = (e: KeyboardEvent) => {
    if (count === 0 || empty) return;
    const from = active ?? (e.key === "ArrowLeft" ? count : -1);
    if (e.key === "ArrowRight") setHover(Math.min(count - 1, from + 1));
    else if (e.key === "ArrowLeft") setHover(Math.max(0, from - 1));
    else if (e.key === "Escape") setHover(null);
    else return;
    e.preventDefault();
  };

  const rows = series
    .map((s, k) => ({ ...s, key: k, value: vals[k][cur] ?? null }))
    .sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
  const flip = x(cur) > 50;

  // The tooltip stays inside the card: it slides back in when its side has too little room
  const rootRef = useRef<HTMLElement>(null);
  const plotRef = useRef<HTMLDivElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const [nudge, setNudge] = useState(0);
  useLayoutEffect(() => {
    const root = rootRef.current;
    const plot = plotRef.current;
    const tip = tipRef.current;
    if (!root || !plot || !tip) return;
    const card = root.getBoundingClientRect();
    const box = plot.getBoundingClientRect();
    const w = tip.offsetWidth;
    const anchor = box.left + (x(cur) / 100) * box.width;
    const left = flip ? anchor - 14 - w : anchor + 14;
    const lo = card.left + 8;
    const hi = card.right - 8 - w;
    setNudge(
      Math.round(
        left < lo ? lo - left : left > hi ? Math.max(lo, hi) - left : 0,
      ),
    );
  });
  const axis = [0, Math.round(count / 2 - 0.5), count - 1].filter(
    (v, i, a) => a.indexOf(v) === i,
  );

  return (
    <>
      <style href="ui-kpi-chart-clean" precedence="default">
        {css}
      </style>
      <section
        className={["ui-kpic", className].filter(Boolean).join(" ")}
        ref={rootRef}
      >
        <div className="ui-kpic__body">
          <h3 className="ui-kpic__title" title={title} dir="auto">
            {title}
          </h3>
          <p className="ui-kpic__value">
            <span className="ui-kpic__amount">{value}</span>
            {finite(delta) && delta !== 0 && (
              <span
                className="ui-kpic__delta"
                data-trend={delta > 0 ? "up" : "down"}
              >
                {delta > 0 ? "+" : "−"}
                {format(Math.abs(delta))}
              </span>
            )}
          </p>

          <div className="ui-kpic__chart">
            <div className="ui-kpic__yaxis" aria-hidden="true">
              {!empty && (
                <>
                  <span>{format(max)}</span>
                  <span>{format((max + min) / 2)}</span>
                  <span>{format(min)}</span>
                </>
              )}
            </div>

            <div
              className="ui-kpic__plot"
              tabIndex={empty ? undefined : 0}
              role="group"
              aria-label={
                empty
                  ? `${title} chart – ${emptyText}`
                  : `${title} chart – use the arrow keys to read the values per day`
              }
              data-empty={empty || undefined}
              ref={plotRef}
              onPointerMove={(e) => setHover(indexAt(e))}
              onPointerDown={(e) => setHover(indexAt(e))}
              onPointerLeave={() => setHover(null)}
              onBlur={() => setHover(null)}
              onKeyDown={onKey}
            >
              <span className="ui-kpic__grid" style={{ top: "0%" }} />
              <span className="ui-kpic__grid" style={{ top: "50%" }} />
              <span className="ui-kpic__grid" style={{ top: "100%" }} />

              {empty && (
                <p className="ui-kpic__empty" title={emptyText}>
                  {emptyText}
                </p>
              )}

              <svg
                className="ui-kpic__lines"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                aria-hidden="true"
              >
                {series.map((s, k) => (
                  <path
                    key={k}
                    d={paths[k]}
                    stroke={s.color}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
              </svg>
              {lone.map((p) => (
                <span
                  key={`${p.k}-${p.i}`}
                  className="ui-kpic__point"
                  style={{
                    left: `${x(p.i)}%`,
                    top: `${y(p.v)}%`,
                    background: series[p.k].color,
                  }}
                  aria-hidden="true"
                />
              ))}

              <div
                className="ui-kpic__marker"
                data-open={active !== null || undefined}
                aria-hidden="true"
              >
                <span
                  className="ui-kpic__guide"
                  style={{ left: `${x(cur)}%` }}
                />
                {series.map((s, k) =>
                  vals[k][cur] === null ? null : (
                    <span
                      key={k}
                      className="ui-kpic__dot"
                      style={{
                        left: `${x(cur)}%`,
                        top: `${y(vals[k][cur] ?? 0)}%`,
                        background: s.color,
                      }}
                    />
                  ),
                )}
              </div>

              <div
                className="ui-kpic__tooltip"
                data-open={active !== null || undefined}
                data-side={flip ? "left" : "right"}
                ref={tipRef}
                style={{
                  left: `${x(cur)}%`,
                  ["--ui-kpic-nudge" as string]: `${nudge}px`,
                }}
                role="status"
                aria-live="polite"
              >
                <p className="ui-kpic__date">
                  {(tooltipLabels ?? labels)[cur]}
                </p>
                <ul>
                  {rows.map((r) => (
                    <li key={r.key}>
                      <span
                        className="ui-kpic__swatch"
                        style={{ background: r.color }}
                      />
                      <span className="ui-kpic__label" dir="auto">
                        {r.label}
                      </span>
                      <span className="ui-kpic__num">
                        {r.value === null ? "—" : format(r.value)}
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div className="ui-kpic__xaxis" aria-hidden="true">
              {axis.map((i) => (
                <span
                  key={i}
                  style={{ left: `${x(i)}%` }}
                  data-edge={
                    i === 0 ? "start" : i === count - 1 ? "end" : undefined
                  }
                >
                  {labels[i]}
                </span>
              ))}
            </div>
          </div>
        </div>

        {footer && (
          <p className="ui-kpic__footer" dir="auto">
            {footer}
          </p>
        )}
      </section>
    </>
  );
}

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-kpic */
.ui-kpic {
  --ui-kpic-card: var(--aui-surface, #ffffff);
  --ui-kpic-pop: var(--aui-overlay, #ffffff);
  --ui-kpic-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-kpic-strong: var(--aui-ink, #000000);
  --ui-kpic-text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-kpic-muted: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-kpic-line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --ui-kpic-grid: rgba(0, 0, 0, 0.06);
  --ui-kpic-guide: rgba(0, 0, 0, 0.12);
  --ui-kpic-up: var(--aui-success, #1e8a4f);
  --ui-kpic-up-bg: color-mix(in srgb, var(--aui-green, #5fc88a) 8%, transparent);
  --ui-kpic-down: var(--aui-red, #f04438);
  --ui-kpic-down-bg: color-mix(in srgb, var(--aui-red, #f04438) 8%, transparent);
  --ui-kpic-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  /* Strokes are shadows: a 0.5px hairline plus a soft lift */
  --ui-kpic-shadow: var(--aui-shadow-card, 0 0 0 0.5px rgba(0, 0, 0, 0.06), 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05));
  --ui-kpic-shadow-pop: var(--aui-shadow-popover, 0 0 0 0.5px rgba(0, 0, 0, 0.08), 0 12px 28px -10px rgba(0, 0, 0, 0.16), 0 1px 2px 0 rgba(0, 0, 0, 0.05));
  --ui-kpic-ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  box-sizing: border-box;
  container-type: inline-size;
  width: 100%;
  max-width: 460px;
  /* One white card – the edge is a shadow, not a border */
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--ui-kpic-card);
  box-shadow: var(--ui-kpic-shadow);
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  color: var(--ui-kpic-ink);
  -webkit-font-smoothing: antialiased;
  font-variant-numeric: tabular-nums;
}
.ui-kpic *, .ui-kpic *::before, .ui-kpic *::after { box-sizing: border-box; }

.ui-kpic__body { padding: 14px 16px 12px; }

/* Title, the big number and the change as a tinted badge */
.ui-kpic__title {
  margin: 0;
  overflow: hidden;
  font-size: 15px;
  font-weight: 500;
  line-height: 24px;
  color: var(--ui-kpic-strong);
  white-space: nowrap;
  text-overflow: ellipsis;
}
.ui-kpic__value {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 8px;
  margin: 6px 0 0;
  font-size: 24px;
  font-weight: 500;
  line-height: 32px;
  letter-spacing: -0.02em;
  color: var(--ui-kpic-strong);
}
.ui-kpic__delta {
  flex: none;
  white-space: nowrap;
  display: inline-flex;
  align-items: center;
  height: 20px;
  padding: 0 6px;
  border-radius: var(--aui-radius-sm, 5px);
  font-size: 12px;
  font-weight: 500;
  line-height: 16px;
  letter-spacing: 0;
}
.ui-kpic__amount { min-width: 0; overflow-wrap: anywhere; }
.ui-kpic__delta[data-trend="up"] { background: var(--ui-kpic-up-bg); color: var(--ui-kpic-up); }
.ui-kpic__delta[data-trend="down"] { background: var(--ui-kpic-down-bg); color: var(--ui-kpic-down); }

/* Chart: y labels on the left, plot on the right, x labels below */
.ui-kpic__chart {
  display: grid;
  grid-template-columns: auto 1fr;
  grid-template-rows: 124px 16px;
  column-gap: 10px;
  row-gap: 8px;
  margin-top: 20px;
}
.ui-kpic__yaxis {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  margin: -8px 0;
  font-size: 11px;
  line-height: 16px;
  text-align: right;
  color: var(--ui-kpic-muted);
}
.ui-kpic__plot {
  position: relative;
  touch-action: pan-y;
  border-radius: 4px;
  outline: none;
  cursor: crosshair;
}
.ui-kpic__plot:focus-visible {
  outline: 2px solid var(--ui-kpic-focus);
  outline-offset: 2px;
}
.ui-kpic__grid {
  position: absolute;
  left: 0;
  right: 0;
  height: 0;
  border-top: 0.5px solid var(--ui-kpic-grid);
}
.ui-kpic__lines {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  overflow: visible;
  animation: ui-kpic-draw 1100ms var(--ui-kpic-ease) both;
}
.ui-kpic__lines path {
  fill: none;
  stroke-width: 1.5;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* Hover marker: a hairline guide and one dot per line – short, crisp moves between points */
.ui-kpic__marker {
  position: absolute;
  inset: 0;
  pointer-events: none;
  opacity: 0;
  transition: opacity 140ms ease;
}
.ui-kpic__marker[data-open] { opacity: 1; }
.ui-kpic__guide,
.ui-kpic__dot {
  position: absolute;
  transition: left 90ms ease-out, top 90ms ease-out;
}
.ui-kpic__guide {
  top: -4px;
  bottom: 0;
  width: 0;
  border-left: 1px solid var(--ui-kpic-guide);
  margin-left: -0.5px;
}
.ui-kpic__dot {
  width: 8px;
  height: 8px;
  margin: -4px 0 0 -4px;
  border-radius: 50%;
  box-shadow: 0 0 0 2px var(--ui-kpic-card);
}

/* A point without neighbours (one version, or a value between gaps) draws no line – it gets a dot */
.ui-kpic__point {
  position: absolute;
  width: 6px;
  height: 6px;
  margin: -3px 0 0 -3px;
  border-radius: 50%;
  box-shadow: 0 0 0 2px var(--ui-kpic-card);
  pointer-events: none;
  animation: ui-kpic-pop 400ms var(--ui-kpic-ease) both;
}

/* Nothing to draw: the grid stays, a quiet line sits on the middle one */
.ui-kpic__plot[data-empty] { cursor: default; }
.ui-kpic__plot[data-empty] > :not(.ui-kpic__grid, .ui-kpic__empty) { display: none; }
.ui-kpic__empty {
  position: absolute;
  top: 50%;
  left: 50%;
  max-width: 100%;
  margin: 0;
  padding: 0 10px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  background: var(--ui-kpic-card);
  transform: translate(-50%, -50%);
  font-size: 12px;
  line-height: 16px;
  text-align: center;
  color: var(--ui-kpic-muted);
}

/* Tooltip: a small white popover beside the guide, flips to the left on the right half */
.ui-kpic__tooltip {
  position: absolute;
  top: -8px;
  z-index: 2;
  min-width: 160px;
  max-width: calc(100cqw - 16px);
  padding: 6px 10px 8px;
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--ui-kpic-pop);
  box-shadow: var(--ui-kpic-shadow-pop);
  pointer-events: none;
  opacity: 0;
  transform: translateX(calc(var(--ui-kpic-shift) + var(--ui-kpic-nudge, 0px))) scale(0.98);
  transition:
    left 90ms ease-out,
    opacity 140ms ease,
    transform 180ms var(--ui-kpic-ease);
}
.ui-kpic__tooltip[data-side="right"] { --ui-kpic-shift: 14px; transform-origin: left top; }
.ui-kpic__tooltip[data-side="left"] { --ui-kpic-shift: calc(-100% - 14px); transform-origin: right top; }
.ui-kpic__tooltip[data-open] {
  opacity: 1;
  transform: translateX(calc(var(--ui-kpic-shift) + var(--ui-kpic-nudge, 0px)));
}
.ui-kpic__date {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  margin: 0;
  font-size: 13px;
  line-height: 20px;
  font-weight: 500;
  color: var(--ui-kpic-strong);
}
.ui-kpic__tooltip ul {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  gap: 0;
  margin: 4px 0 0;
  padding: 4px 0 0;
  border-top: 0.5px solid var(--ui-kpic-line);
  list-style: none;
}
.ui-kpic__tooltip li {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  line-height: 20px;
  color: var(--ui-kpic-text);
}
.ui-kpic__swatch {
  flex: none;
  width: 8px;
  height: 8px;
  border-radius: 2px;
}
.ui-kpic__label {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.ui-kpic__num {
  flex: none;
  padding-left: 16px;
  color: var(--ui-kpic-ink);
}

.ui-kpic__xaxis {
  position: relative;
  grid-column: 2;
  font-size: 11px;
  line-height: 16px;
  color: var(--ui-kpic-muted);
}
.ui-kpic__xaxis span {
  position: absolute;
  top: 0;
  max-width: 34%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  transform: translateX(-50%);
}
.ui-kpic__xaxis span[data-edge="start"] { transform: none; }
.ui-kpic__xaxis span[data-edge="end"] { transform: translateX(-100%); }

/* The last row of the card */
.ui-kpic__footer {
  overflow-wrap: anywhere;
  margin: 0;
  padding: 10px 16px;
  border-top: 0.5px solid var(--ui-kpic-line);
  font-size: 12px;
  line-height: 16px;
  color: var(--ui-kpic-muted);
}

@keyframes ui-kpic-draw {
  from { clip-path: inset(-10% 100% -10% 0); }
  to { clip-path: inset(-10% -2% -10% 0); }
}
@keyframes ui-kpic-pop {
  from { opacity: 0; transform: scale(0.4); }
}

/* Narrow (phones): tighter padding, a slightly shorter plot */
@container (max-width: 360px) {
  .ui-kpic__body { padding: 12px 12px 10px; }
  .ui-kpic__chart { grid-template-rows: 112px 16px; column-gap: 8px; }
  .ui-kpic__footer { padding: 10px 12px; }
  .ui-kpic__tooltip { min-width: 144px; }
  .ui-kpic__num { padding-left: 12px; }
}

@media (prefers-reduced-motion: reduce) {
  .ui-kpic__lines, .ui-kpic__point { animation: none; }
  .ui-kpic__marker, .ui-kpic__guide, .ui-kpic__dot, .ui-kpic__tooltip { transition: none; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-kpic {
  --ui-kpic-card: var(--aui-surface, #1c1c1e);
  --ui-kpic-pop: var(--aui-overlay, #252527);
  --ui-kpic-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-kpic-strong: var(--aui-ink, #ffffff);
  --ui-kpic-text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-kpic-muted: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-kpic-line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --ui-kpic-grid: rgba(255, 255, 255, 0.08);
  --ui-kpic-guide: rgba(255, 255, 255, 0.2);
  --ui-kpic-up: var(--aui-green, #5fc88a);
  --ui-kpic-up-bg: color-mix(in srgb, var(--aui-green, #7dcf9f) 14%, transparent);
  --ui-kpic-down: var(--aui-red, #ff6b5f);
  --ui-kpic-down-bg: color-mix(in srgb, var(--aui-red, #ff6b5f) 14%, transparent);
  --ui-kpic-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
  --ui-kpic-shadow: var(--aui-shadow-card, 0 0 0 0.5px rgba(255, 255, 255, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.4));
  --ui-kpic-shadow-pop: var(--aui-shadow-popover, 0 0 0 0.5px rgba(255, 255, 255, 0.12), 0 12px 28px -10px rgba(0, 0, 0, 0.6));
}
`;
