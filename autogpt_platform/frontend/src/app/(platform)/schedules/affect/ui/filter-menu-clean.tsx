"use client";

import {
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type KeyboardEvent,
  type RefObject,
} from "react";

/*
 * Filter Menu (Clean) – grouped filters with chips, in a single self-contained file.
 * Clean style: no tray – one white card with hairline shadows, gray tab-like chips on one line ("+N" for the rest),
 * quiet rows with a subtle hover, blue checkboxes and the keyboard hints as the card's last row.
 * Active filters sit on top as removable chips; below, collapsible groups list their options as checkboxes
 * with a count of what's picked. Arrow keys move through groups and options, Enter or Space picks,
 * ← and → fold a group. Heights glide when groups open or chips come and go.
 * The styles are injected below, so no extra files are needed.
 */

export type FilterOption = { id: string; label: string };

export type FilterGroup = {
  id: string;
  label: string;
  options: FilterOption[];
  /** Put in front of the option on its chip, e.g. "Score: " → "Score: 80–90" */
  chipPrefix?: string;
};

/** Picked option ids per group id */
export type FilterValue = Record<string, string[]>;

export type FilterMenuProps = {
  groups: FilterGroup[];
  /** Controlled value */
  value?: FilterValue;
  /** Starting value when uncontrolled */
  defaultValue?: FilterValue;
  onChange?: (value: FilterValue) => void;
  /** Groups that start open – by default the first one */
  defaultOpen?: string[];
  title?: string;
  /** Show the keyboard hints as the last row of the card */
  hints?: boolean;
  className?: string;
};

type Row =
  | { kind: "group"; group: FilterGroup; gi: number }
  | {
      kind: "option";
      group: FilterGroup;
      option: FilterOption;
      gi: number;
      oi: number;
    };

/** Gap between chips, in px – matches the CSS */
const CHIP_GAP = 4;

export function FilterMenu({
  groups: rawGroups,
  value,
  defaultValue = {},
  onChange,
  defaultOpen,
  title = "Filters",
  hints = true,
  className,
}: FilterMenuProps) {
  // Gaps, groups without an options list and options without a label are skipped; ids stay as given
  const groups = (Array.isArray(rawGroups) ? rawGroups : [])
    .filter((g): g is FilterGroup => !!g && typeof g === "object")
    .map((g) => ({
      ...g,
      label: String(g.label ?? ""),
      options: (Array.isArray(g.options) ? g.options : [])
        .filter((o): o is FilterOption => !!o && typeof o === "object")
        .map((o) => ({ ...o, label: String(o.label ?? "") })),
    }));
  const uid = useId();
  const [inner, setInner] = useState<FilterValue>(defaultValue);
  const picked = value ?? inner;
  const [open, setOpen] = useState<Set<string>>(
    () => new Set(defaultOpen ?? groups.slice(0, 1).map((g) => g.id)),
  );
  const [focus, setFocus] = useState<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const chipsInner = useRef<HTMLDivElement>(null);
  const chipsOuter = useSmoothHeight(chipsInner);
  const measure = useRef<HTMLDivElement>(null);
  const [fit, setFit] = useState(Infinity);

  const commit = (next: FilterValue) => {
    if (value === undefined) setInner(next);
    onChange?.(next);
  };
  // Only lists count – a string or a stray value in `value` picks nothing
  const pickedIn = (g: string) => (Array.isArray(picked[g]) ? picked[g] : []);
  const isOn = (g: string, o: string) => pickedIn(g).includes(o);
  const toggle = (g: string, o: string) => {
    const cur = pickedIn(g);
    commit({
      ...picked,
      [g]: cur.includes(o) ? cur.filter((x) => x !== o) : [...cur, o],
    });
  };
  const toggleGroup = (g: string) =>
    setOpen((s) => {
      const next = new Set(s);
      if (next.has(g)) next.delete(g);
      else next.add(g);
      return next;
    });

  // Chips in the order of the groups and their options – one per picked id, even if an id repeats
  const seen = new Set<string>();
  const chips = groups.flatMap((g) =>
    g.options
      .filter(
        (o) =>
          isOn(g.id, o.id) &&
          !seen.has(`${g.id}\u0000${o.id}`) &&
          !!seen.add(`${g.id}\u0000${o.id}`),
      )
      .map((o) => ({
        group: g,
        option: o,
        label: `${g.chipPrefix ?? ""}${o.label}`,
      })),
  );
  const chipKey = chips
    .map((c) => `${c.group.id}:${c.option.id}:${c.label}`)
    .join("|");

  // One line only: as many chips as fit, the rest hide behind "+N".
  // A hidden copy of the chips (plus a "+99" sample) is measured to know their widths before they are placed.
  useLayoutEffect(() => {
    const row = chipsInner.current;
    const list = measure.current;
    if (!row || !list) return;
    const place = () => {
      const widths = [...list.children].map(
        (el) => (el as HTMLElement).offsetWidth + CHIP_GAP,
      );
      const more = widths.pop() ?? 0;
      const style = getComputedStyle(row);
      const room =
        row.clientWidth -
        parseFloat(style.paddingLeft) -
        parseFloat(style.paddingRight) +
        CHIP_GAP;
      let used = 0;
      let n = 0;
      for (; n < widths.length; n++) {
        const rest = n + 1 < widths.length ? more : 0;
        if (used + widths[n] + rest > room) break;
        used += widths[n];
      }
      setFit(n);
    };
    place();
    const ro = new ResizeObserver(place);
    ro.observe(row);
    return () => ro.disconnect();
  }, [chipKey]);

  const shown = chips.slice(0, fit);
  const hidden = chips.slice(fit);

  // Everything the arrow keys can reach, top to bottom
  // Keys come from positions, so groups or options that share an id stay reachable
  const rows: Row[] = groups.flatMap((g, gi) => [
    { kind: "group", group: g, gi } as Row,
    ...(open.has(g.id)
      ? g.options.map(
          (o, oi) => ({ kind: "option", group: g, option: o, gi, oi }) as Row,
        )
      : []),
  ]);
  const keyOf = (r: Row) =>
    r.kind === "group" ? `g:${r.gi}` : `o:${r.gi}:${r.oi}`;

  const moveTo = (key: string) => {
    setFocus(key);
    listRef.current
      ?.querySelector<HTMLElement>(`[data-key="${CSS.escape(key)}"]`)
      ?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    const i = rows.findIndex((r) => keyOf(r) === focus);
    const row = rows[i];
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      const next =
        rows[
          Math.max(
            0,
            Math.min(rows.length - 1, i + (e.key === "ArrowDown" ? 1 : -1)),
          )
        ] ?? rows[0];
      moveTo(keyOf(next));
    } else if (e.key === "Home" || e.key === "End") {
      e.preventDefault();
      moveTo(keyOf(rows[e.key === "Home" ? 0 : rows.length - 1]));
    } else if (row && (e.key === "ArrowRight" || e.key === "ArrowLeft")) {
      e.preventDefault();
      const g = row.group.id;
      if (e.key === "ArrowRight" && !open.has(g)) toggleGroup(g);
      if (e.key === "ArrowLeft") {
        if (open.has(g)) toggleGroup(g);
        moveTo(`g:${row.gi}`);
      }
    } else if (row && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      if (row.kind === "group") toggleGroup(row.group.id);
      else toggle(row.group.id, row.option.id);
    }
  };

  return (
    <div className={["ui-fmc", className].filter(Boolean).join(" ")}>
      {/* React 19 hoists this into <head> once, no matter how many menus render */}
      <style href="ui-filter-menu-clean" precedence="default">
        {css}
      </style>

      <div className="ui-fmc__head">
        <span className="ui-fmc__title" dir="auto" title={title}>
          {title}
        </span>
        <button
          type="button"
          className="ui-fmc__clear"
          disabled={chips.length === 0}
          onClick={() => commit({})}
        >
          Clear all
        </button>
      </div>

      {/* Active filters on one line – the box glides to its new height as the line appears or goes */}
      <div className="ui-fmc__chips-outer" ref={chipsOuter}>
        <div
          className="ui-fmc__chips"
          ref={chipsInner}
          data-empty={chips.length === 0 || undefined}
          aria-live="polite"
        >
          {/* Hidden copy, only to measure how many chips fit on the line */}
          <div className="ui-fmc__measure" ref={measure} aria-hidden="true">
            {chips.map((c, i) => (
              <span key={i} className="ui-fmc__chip">
                <span className="ui-fmc__chip-label">{c.label}</span>
                <span className="ui-fmc__chip-x" />
              </span>
            ))}
            <span className="ui-fmc__chip ui-fmc__more">+99</span>
          </div>
          {shown.map((c, i) => (
            <span key={i} className="ui-fmc__chip">
              <span className="ui-fmc__chip-label" dir="auto" title={c.label}>
                {c.label}
              </span>
              <button
                type="button"
                className="ui-fmc__chip-x"
                aria-label={`Remove ${c.label}`}
                onClick={() => toggle(c.group.id, c.option.id)}
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />
                </svg>
              </button>
            </span>
          ))}
          {hidden.length > 0 && (
            <span
              className="ui-fmc__chip ui-fmc__more"
              title={hidden.map((c) => c.label).join(", ")}
              aria-label={`${hidden.length} more: ${hidden.map((c) => c.label).join(", ")}`}
            >
              <span key={hidden.length} className="ui-fmc__roll">
                +{hidden.length}
              </span>
            </span>
          )}
        </div>
      </div>

      <div className="ui-fmc__groups" ref={listRef} onKeyDown={onKeyDown}>
        {groups.map((g, gi) => {
          const isOpen = open.has(g.id);
          const count = new Set(
            pickedIn(g.id).filter((id) => g.options.some((o) => o.id === id)),
          ).size;
          const gKey = `g:${gi}`;
          return (
            <section
              key={`${g.id}#${gi}`}
              className="ui-fmc__group"
              data-open={isOpen ? "" : undefined}
            >
              <button
                type="button"
                className="ui-fmc__group-head"
                data-key={gKey}
                tabIndex={
                  focus === gKey || (focus === null && gi === 0) ? 0 : -1
                }
                aria-expanded={isOpen}
                aria-controls={`${uid}-${gi}`}
                onFocus={() => setFocus(gKey)}
                onClick={() => toggleGroup(g.id)}
              >
                <svg
                  className="ui-fmc__chevron"
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                >
                  <path d="M4.5 6.25 8 9.75l3.5-3.5" />
                </svg>
                <span
                  className="ui-fmc__group-label"
                  dir="auto"
                  title={g.label}
                >
                  {g.label}
                </span>
                <span className="ui-fmc__group-count">
                  {g.options.length.toLocaleString("en-US")}
                </span>
                <span className="ui-fmc__group-picked" aria-live="polite">
                  {count > 0 && (
                    <span key={count} className="ui-fmc__roll">
                      {`${count} filter${count === 1 ? "" : "s"}`}
                    </span>
                  )}
                </span>
              </button>
              {/* Opens and closes by animating its grid row from 0 to the content height */}
              <div
                className="ui-fmc__panel"
                id={`${uid}-${gi}`}
                role="group"
                aria-label={g.label}
              >
                <div className="ui-fmc__panel-inner">
                  {g.options.map((o, oi) => {
                    const key = `o:${gi}:${oi}`;
                    const on = isOn(g.id, o.id);
                    return (
                      <button
                        key={`${o.id}#${oi}`}
                        type="button"
                        role="checkbox"
                        aria-checked={on}
                        className="ui-fmc__option"
                        data-key={key}
                        tabIndex={isOpen && focus === key ? 0 : -1}
                        onFocus={() => setFocus(key)}
                        onClick={() => toggle(g.id, o.id)}
                      >
                        <span className="ui-fmc__box" aria-hidden="true">
                          <svg viewBox="0 0 16 16">
                            <path d="m4.25 8.25 2.5 2.5 5-5.5" />
                          </svg>
                        </span>
                        <span
                          className="ui-fmc__option-label"
                          dir="auto"
                          title={o.label}
                        >
                          {o.label}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </section>
          );
        })}
      </div>

      {/* The last row of the card, under a hairline */}
      {hints && (
        <div className="ui-fmc__hints" aria-hidden="true">
          <span className="ui-fmc__hint">
            <kbd>↑</kbd>
            <kbd>↓</kbd>
            <span>to navigate</span>
          </span>
          <span className="ui-fmc__hint">
            <kbd>↵</kbd>
            <span>to select</span>
          </span>
        </div>
      )}
    </div>
  );
}

/** The outer box follows the inner box's height, with a transition after the first frame */
function useSmoothHeight(inner: RefObject<HTMLElement | null>) {
  const outer = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    const o = outer.current;
    const i = inner.current;
    if (!o || !i) return;
    const sync = () => {
      o.style.height = `${i.offsetHeight}px`;
    };
    sync();
    const ro = new ResizeObserver(sync);
    ro.observe(i);
    const frame = requestAnimationFrame(() =>
      o.setAttribute("data-smooth", ""),
    );
    return () => {
      ro.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [inner]);
  return outer;
}

/* ---------- Styles ---------- */

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-fmc */
.ui-fmc {
  --ui-fmc-card: #ffffff;
  --ui-fmc-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-fmc-strong: var(--aui-ink, #000000);
  --ui-fmc-text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-fmc-muted: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-fmc-faint: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --ui-fmc-icon: var(--aui-muted, rgba(0, 0, 0, 0.447));
  --ui-fmc-line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --ui-fmc-fill: rgba(0, 0, 0, 0.06);
  --ui-fmc-fill-strong: rgba(0, 0, 0, 0.1);
  --ui-fmc-wash: var(--aui-subtle, rgba(0, 0, 0, 0.03));
  --ui-fmc-box-ring: var(--aui-border-strong, rgba(0, 0, 0, 0.22));
  --ui-fmc-accent: rgba(0, 96, 226, 0.765);
  --ui-fmc-check: #ffffff;
  --ui-fmc-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  /* Strokes are shadows: a 0.5px hairline plus a soft lift */
  --ui-fmc-shadow: 0 0 0 0.5px rgba(0, 0, 0, 0.06), 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --ui-fmc-ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  --ui-fmc-spring: cubic-bezier(0.34, 1.56, 0.64, 1);
  box-sizing: border-box;
  width: 100%;
  max-width: 400px;
  container-type: inline-size;
  overflow: hidden;
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--aui-surface, var(--ui-fmc-card));
  box-shadow: var(--aui-shadow-card, var(--ui-fmc-shadow));
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  color: var(--ui-fmc-ink);
  -webkit-font-smoothing: antialiased;
}
.ui-fmc *, .ui-fmc *::before, .ui-fmc *::after { box-sizing: border-box; }
:where(.ui-fmc) button { margin: 0; padding: 0; border: 0; background: none; font: inherit; color: inherit; text-align: left; cursor: pointer; -webkit-tap-highlight-color: transparent; }
:where(.ui-fmc) svg { display: block; flex: none; }
.ui-fmc button:focus-visible { outline: 2px solid var(--ui-fmc-focus); outline-offset: 2px; }

/* ---------- Header ---------- */

.ui-fmc__head { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 10px 8px 16px; }
.ui-fmc__title { min-width: 0; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; text-align: start; font-size: 15px; line-height: 24px; font-weight: 500; color: var(--ui-fmc-strong); }
/* Clear all: a quiet ghost button */
.ui-fmc__clear {
  flex: none;
  height: 28px;
  padding: 0 8px !important;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  font-size: 13px;
  font-weight: 500;
  color: var(--ui-fmc-text);
  transition: color 160ms ease, background-color 160ms ease;
}
.ui-fmc__clear:hover:not(:disabled) { background: var(--ui-fmc-wash); color: var(--ui-fmc-ink); }
.ui-fmc__clear:disabled { color: var(--ui-fmc-faint); cursor: default; }

/* ---------- Chips: one line, the rest go into "+N" ---------- */

.ui-fmc__chips-outer { overflow: hidden; }
.ui-fmc__chips-outer[data-smooth] { transition: height 320ms var(--ui-fmc-ease); }
.ui-fmc__chips { position: relative; display: flex; flex-wrap: nowrap; gap: 4px; padding: 0 16px 12px; overflow: hidden; }
.ui-fmc__chips[data-empty] { padding-bottom: 0; }
.ui-fmc__measure { position: absolute; top: 0; left: 0; display: flex; gap: 4px; width: max-content; visibility: hidden; pointer-events: none; }
.ui-fmc__chip {
  display: inline-flex;
  flex: none;
  align-items: center;
  gap: 2px;
  height: 24px;
  padding: 0 3px 0 8px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  background: var(--aui-subtle, var(--ui-fmc-fill));
  font-size: 12px;
  line-height: 16px;
  font-weight: 500;
  color: var(--ui-fmc-ink);
  white-space: nowrap;
  animation: ui-fmc-chip-in 260ms var(--ui-fmc-ease) both;
}
.ui-fmc__chip-label { min-width: 0; max-width: 180px; overflow: hidden; text-overflow: ellipsis; }
.ui-fmc__chip-x {
  display: grid;
  place-items: center;
  flex: none;
  width: 18px;
  height: 18px;
  border-radius: calc(var(--aui-radius-sm, 5px) - 1px);
  color: var(--ui-fmc-icon);
  transition: background-color 140ms ease, color 140ms ease;
}
.ui-fmc__chip-x:hover { background: var(--ui-fmc-fill); color: var(--ui-fmc-ink); }
.ui-fmc__chip-x svg { width: 12px; height: 12px; fill: none; stroke: currentColor; stroke-width: 1.4; stroke-linecap: round; }
.ui-fmc__more { padding: 0 8px; color: var(--ui-fmc-text); font-variant-numeric: tabular-nums; cursor: default; }
.ui-fmc__roll { display: inline-block; animation: ui-fmc-roll 300ms var(--ui-fmc-ease) both; }

/* ---------- Groups ---------- */

.ui-fmc__groups { display: grid; grid-template-columns: minmax(0, 1fr); padding: 4px; border-top: 0.5px solid var(--ui-fmc-line); }
.ui-fmc__group-head {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 32px;
  padding: 0 8px !important;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  /* Hover comes in crisp and fades out when the pointer moves on */
  transition: background-color 140ms ease;
}
.ui-fmc__group-head:hover { background: var(--ui-fmc-wash); transition-duration: 90ms; }
.ui-fmc__group-head:focus-visible, .ui-fmc__option:focus-visible { outline-offset: -2px; }
.ui-fmc__chevron {
  width: 16px;
  height: 16px;
  fill: none;
  stroke: var(--ui-fmc-icon);
  stroke-width: 1.1;
  stroke-linecap: round;
  stroke-linejoin: round;
  transform: rotate(-90deg);
  transition: transform 260ms var(--ui-fmc-ease);
}
.ui-fmc__group[data-open] .ui-fmc__chevron { transform: none; }
.ui-fmc__group-label { min-width: 0; overflow: hidden; text-overflow: ellipsis; text-align: start; font-weight: 500; color: var(--ui-fmc-ink); white-space: nowrap; }
.ui-fmc__group-count { flex: none; font-size: 12px; line-height: 16px; color: var(--ui-fmc-faint); font-variant-numeric: tabular-nums; }
.ui-fmc__group-picked { display: inline-flex; flex: none; margin-left: auto; overflow: hidden; font-size: 12px; line-height: 16px; color: var(--ui-fmc-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }

.ui-fmc__panel {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 320ms var(--ui-fmc-ease);
}
.ui-fmc__group[data-open] .ui-fmc__panel { grid-template-rows: 1fr; }
.ui-fmc__panel-inner {
  display: grid;
  grid-template-columns: minmax(0, 1fr);
  min-height: 0;
  overflow: hidden;
  padding: 0;
  opacity: 0;
  transition: opacity 220ms ease, padding 320ms var(--ui-fmc-ease);
}
.ui-fmc__group[data-open] .ui-fmc__panel-inner { opacity: 1; padding: 0 0 4px; }
/* A long list scrolls inside the open group, so the card keeps a sane height */
.ui-fmc__group[data-open] .ui-fmc__panel-inner { max-height: 320px; overflow-x: hidden; overflow-y: auto; overscroll-behavior: contain; }

/* Options sit under the group's label, the checkbox lined up with its text */
.ui-fmc__option {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  height: 32px;
  padding: 0 8px 0 32px !important;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  color: var(--ui-fmc-ink);
  transition: background-color 140ms ease;
}
.ui-fmc__option:hover { background: var(--ui-fmc-wash); transition-duration: 90ms; }
.ui-fmc__option-label { min-width: 0; text-align: start; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
/* Checkbox: 14px with a hairline, blue when picked; the check draws itself in */
.ui-fmc__box {
  display: grid;
  place-items: center;
  flex: none;
  width: 14px;
  height: 14px;
  border-radius: 4px;
  background: var(--ui-fmc-card);
  box-shadow: inset 0 0 0 0.5px var(--ui-fmc-box-ring), 0 1px 1px 0 rgba(0, 0, 0, 0.04);
  transition: background-color 160ms ease, box-shadow 160ms ease;
}
.ui-fmc__box svg {
  width: 12px;
  height: 12px;
  fill: none;
  stroke: var(--ui-fmc-check);
  stroke-width: 1.6;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 12;
  stroke-dashoffset: 12;
  transition: stroke-dashoffset 240ms var(--ui-fmc-ease);
}
.ui-fmc__option[aria-checked="true"] .ui-fmc__box { background: var(--ui-fmc-accent); box-shadow: none; }
.ui-fmc__option[aria-checked="true"] .ui-fmc__box svg { stroke-dashoffset: 0; transition-delay: 40ms; }

/* ---------- Keyboard hints: the last row of the card ---------- */

.ui-fmc__hints {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  height: 32px;
  padding: 0 12px;
  border-top: 0.5px solid var(--ui-fmc-line);
  font-size: 12px;
  line-height: 16px;
  color: var(--ui-fmc-muted);
}
.ui-fmc__hint { display: inline-flex; align-items: center; gap: 2px; white-space: nowrap; }
.ui-fmc__hint > span { margin-left: 4px; }
.ui-fmc kbd {
  display: inline-grid;
  place-items: center;
  min-width: 16px;
  height: 16px;
  padding: 0 4px;
  border-radius: calc(var(--aui-radius-sm, 5px) - 1px);
  background: var(--ui-fmc-card);
  box-shadow: var(--ui-fmc-shadow);
  font-family: inherit;
  font-size: 10px;
  line-height: 16px;
  font-weight: 500;
  color: var(--ui-fmc-text);
}

@keyframes ui-fmc-chip-in { from { opacity: 0; transform: scale(0.9); } }
@keyframes ui-fmc-roll { from { opacity: 0; transform: translateY(60%); } to { opacity: 1; transform: none; } }

/* Narrow: a bit less side padding, shorter chips */
@container (max-width: 320px) {
  .ui-fmc__head { padding-left: 14px; }
  .ui-fmc__chips { padding-left: 14px; padding-right: 14px; }
  .ui-fmc__chip-label { max-width: 132px; }
}
/* Touch screens: taller rows and buttons */
@media (hover: none) and (pointer: coarse) {
  .ui-fmc__clear { height: 36px; }
  .ui-fmc__chip { height: 32px; }
  .ui-fmc__chip-x { width: 26px; height: 26px; }
  .ui-fmc__group-head { height: 44px; }
  .ui-fmc__option { height: 44px; }
  .ui-fmc__box { width: 18px; height: 18px; border-radius: 5px; }
  .ui-fmc__box svg { width: 14px; height: 14px; }
  .ui-fmc__hints { display: none; }
}
@media (prefers-reduced-motion: reduce) {
  .ui-fmc *, .ui-fmc *::before, .ui-fmc *::after { animation: none !important; transition: none !important; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-fmc {
  --ui-fmc-card: #1c1c1e;
  --ui-fmc-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-fmc-strong: var(--aui-ink, #ffffff);
  --ui-fmc-text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-fmc-muted: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-fmc-faint: var(--aui-placeholder, rgba(255, 255, 255, 0.3));
  --ui-fmc-icon: var(--aui-muted, rgba(255, 255, 255, 0.5));
  --ui-fmc-line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --ui-fmc-fill: rgba(255, 255, 255, 0.08);
  --ui-fmc-fill-strong: rgba(255, 255, 255, 0.13);
  --ui-fmc-wash: var(--aui-subtle, rgba(255, 255, 255, 0.04));
  --ui-fmc-box-ring: var(--aui-border-strong, rgba(255, 255, 255, 0.28));
  --ui-fmc-accent: var(--aui-blue, #4fb0fe);
  --ui-fmc-check: #ffffff;
  --ui-fmc-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
  --ui-fmc-shadow: 0 0 0 0.5px rgba(255, 255, 255, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.4);
}
`;
