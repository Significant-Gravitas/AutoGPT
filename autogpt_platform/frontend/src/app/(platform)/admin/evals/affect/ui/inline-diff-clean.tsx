"use client";

import {
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent,
  type Ref,
} from "react";

/*
 * Inline Diff (Clean) – review an AI's edits to a piece of prose, in a single self-contained file.
 * Clean style: no tray – one white card with hairline shadows; edits are quiet tinted fills without strokes.
 * The original and the suggested text are diffed word by word and shown as one flowing paragraph:
 * removed words struck through on a soft red tint, new words on a soft green one. Hover or focus a
 * change and a small popover offers Accept and Reject; accepting shrinks the old words away and
 * settles the new ones into the text, rejecting does the opposite. Tab moves between changes,
 * Y / Enter accepts, N / Backspace rejects. A toolbar counts what's left, steps to the previous /
 * next pending change with its arrows and turns into "All changes reviewed" with an Undo at the end.
 * The optional label opens the card, the toolbar is its last row under a hairline.
 * The styles are injected below, so no extra files are needed.
 */

/** A run of unchanged text, or one change (removed and/or inserted text) */
export type DiffSegment =
  | { type: "equal"; text: string }
  | { type: "change"; remove: string; insert: string };

export type InlineDiffVerdict = "accepted" | "rejected";

/** Imperative controls, e.g. for a guided tour or your own buttons */
export type InlineDiffHandle = {
  /** Show the popover on a change (by its index among the changes) without moving focus; null hides it */
  highlight: (index: number | null) => void;
  accept: (index: number) => void;
  reject: (index: number) => void;
  acceptAll: () => void;
  rejectAll: () => void;
  /** Restores the last decision (or the last "all" batch) */
  undo: () => void;
  /** Back to every change pending */
  reset: () => void;
};

export type InlineDiffProps = {
  /** The text as it is now */
  original?: string;
  /** The text the AI suggests */
  suggested?: string;
  /** Pre-computed segments instead of original + suggested */
  changes?: DiffSegment[];
  /** Called on every decision (and undo) with the text so far – pending changes keep the original words */
  onChange?: (text: string) => void;
  /** Called once no change is pending, with the final text */
  onDone?: (text: string) => void;
  /** Small line at the top of the card, e.g. "Suggested by AI · Make it shorter" */
  label?: string;
  ref?: Ref<InlineDiffHandle>;
  className?: string;
};

/* ---------- Word diff ---------- */

// Words (letters, marks, digits, apostrophes), runs of whitespace, whole emoji (skin tones, ZWJ families, flags) and single punctuation marks
const EMOJI =
  "\\p{Extended_Pictographic}[\\u{1F3FB}-\\u{1F3FF}\\uFE0F\\u20E3]*(?:\\u200D\\p{Extended_Pictographic}[\\u{1F3FB}-\\u{1F3FF}\\uFE0F]*)*";
const TOKEN = new RegExp(
  `\\s+|[\\p{L}\\p{M}\\p{N}_'’]+|${EMOJI}|\\p{Regional_Indicator}{2}|[^\\s\\p{L}\\p{N}_]\\p{M}*`,
  "gu",
);
const isSpace = (s: string) => /^\s+$/.test(s);

/** Shortest edit script (Myers, O((n+m)·d)) – null when it needs more than `limit` edits */
function myers(
  x: string[],
  y: string[],
  limit = 1500,
): { t: 0 | 1 | 2; s: string }[] | null {
  const n = x.length;
  const m = y.length;
  const off = n + m + 1;
  const v = new Int32Array(2 * off + 1);
  const trace: Int32Array[] = [];
  let found = -1;
  for (let d = 0; d <= Math.min(limit, n + m) && found < 0; d++) {
    // v as it was before step d: index k sits at k + d + 1
    trace.push(v.slice(off - d - 1, off + d + 2));
    for (let k = -d; k <= d; k += 2) {
      let i =
        k === -d || (k !== d && v[off + k - 1] < v[off + k + 1])
          ? v[off + k + 1]
          : v[off + k - 1] + 1;
      let j = i - k;
      while (i < n && j < m && x[i] === y[j]) {
        i++;
        j++;
      }
      v[off + k] = i;
      if (i >= n && j >= m) {
        found = d;
        break;
      }
    }
  }
  if (found < 0) return null;
  const rev: { t: 0 | 1 | 2; s: string }[] = [];
  let i = n;
  let j = m;
  for (let d = found; d > 0; d--) {
    const t = trace[d];
    const at = (k: number) => t[k + d + 1];
    const k = i - j;
    const down = k === -d || (k !== d && at(k - 1) < at(k + 1));
    const pk = down ? k + 1 : k - 1;
    const pi = at(pk);
    const pj = pi - pk;
    while (i > pi && j > pj) (rev.push({ t: 0, s: x[--i] }), j--);
    if (down) rev.push({ t: 2, s: y[--j] });
    else rev.push({ t: 1, s: x[--i] });
  }
  while (i > 0 && j > 0) (rev.push({ t: 0, s: x[--i] }), j--);
  return rev.reverse();
}

/** Word-level diff of two texts (LCS over word and whitespace tokens), grouped into equal runs and changes */
export function diffWords(original: string, suggested: string): DiffSegment[] {
  const a = original.match(TOKEN) ?? [];
  const b = suggested.match(TOKEN) ?? [];

  // Strip the common start and end so the table only covers the middle
  let pre = 0;
  while (pre < a.length && pre < b.length && a[pre] === b[pre]) pre++;
  let suf = 0;
  while (
    suf < a.length - pre &&
    suf < b.length - pre &&
    a[a.length - 1 - suf] === b[b.length - 1 - suf]
  )
    suf++;
  const x = a.slice(pre, a.length - suf);
  const y = b.slice(pre, b.length - suf);
  const n = x.length;
  const m = y.length;

  type Op = { t: 0 | 1 | 2; s: string }; // 0 equal, 1 removed, 2 inserted
  const ops: Op[] = a.slice(0, pre).map((s) => ({ t: 0, s }));

  if (n * m > 4_000_000) {
    // Long texts: Myers' diff, fast when the edits are few; only a rewrite of everything falls back to one replacement
    const mid = myers(x, y);
    if (mid) ops.push(...mid);
    else {
      x.forEach((s) => ops.push({ t: 1, s }));
      y.forEach((s) => ops.push({ t: 2, s }));
    }
  } else {
    const w = m + 1;
    const L = new Uint16Array((n + 1) * w);
    for (let i = n - 1; i >= 0; i--)
      for (let j = m - 1; j >= 0; j--)
        L[i * w + j] =
          x[i] === y[j]
            ? L[(i + 1) * w + j + 1] + 1
            : Math.max(L[(i + 1) * w + j], L[i * w + j + 1]);
    let i = 0;
    let j = 0;
    while (i < n && j < m) {
      if (x[i] === y[j]) {
        ops.push({ t: 0, s: x[i] });
        i++;
        j++;
      } else if (L[(i + 1) * w + j] >= L[i * w + j + 1])
        ops.push({ t: 1, s: x[i++] });
      else ops.push({ t: 2, s: y[j++] });
    }
    while (i < n) ops.push({ t: 1, s: x[i++] });
    while (j < m) ops.push({ t: 2, s: y[j++] });
  }
  a.slice(a.length - suf).forEach((s) => ops.push({ t: 0, s }));

  // A pure removal or insertion that starts with a space slides right, so the space trails the words
  for (let s = 0; s < ops.length; s++) {
    if (ops[s].t === 0 || (s > 0 && ops[s - 1].t !== 0)) continue;
    const t = ops[s].t;
    let e = s;
    while (e < ops.length && ops[e].t === t) e++;
    if (e < ops.length && ops[e].t !== 0) continue; // mixed change – leave it
    while (
      e < ops.length &&
      ops[e].t === 0 &&
      ops[e].s === ops[s].s &&
      isSpace(ops[s].s)
    ) {
      ops[e] = { t, s: ops[e].s };
      ops[s] = { t: 0, s: ops[s].s };
      s++;
      e++;
    }
    s = e - 1;
  }

  // Group into segments
  const raw: DiffSegment[] = [];
  for (const op of ops) {
    const last = raw[raw.length - 1];
    if (op.t === 0) {
      if (last?.type === "equal") last.text += op.s;
      else raw.push({ type: "equal", text: op.s });
    } else {
      if (last?.type !== "change")
        raw.push({ type: "change", remove: "", insert: "" });
      const c = raw[raw.length - 1] as Extract<DiffSegment, { type: "change" }>;
      if (op.t === 1) c.remove += op.s;
      else c.insert += op.s;
    }
  }

  // Changes split only by a space read as one edit: "a lot faster" → "twice as fast"
  const merged: DiffSegment[] = [];
  for (let k = 0; k < raw.length; k++) {
    const seg = raw[k];
    const prev = merged[merged.length - 1];
    const next = raw[k + 1];
    if (
      seg.type === "equal" &&
      isSpace(seg.text) &&
      prev?.type === "change" &&
      next?.type === "change"
    ) {
      prev.remove += seg.text + next.remove;
      prev.insert += seg.text + next.insert;
      k++;
      continue;
    }
    merged.push({ ...seg });
  }

  // Whitespace both sides share at their edges belongs to the surrounding text
  const out: DiffSegment[] = [];
  const pushEqual = (text: string) => {
    if (!text) return;
    const last = out[out.length - 1];
    if (last?.type === "equal") last.text += text;
    else out.push({ type: "equal", text });
  };
  for (const seg of merged) {
    if (seg.type === "equal") {
      pushEqual(seg.text);
      continue;
    }
    let { remove, insert } = seg;
    let lead = "";
    let trail = "";
    if (remove && insert) {
      const l = remove.match(/^\s+/)?.[0] ?? "";
      if (
        l &&
        insert.startsWith(l) &&
        l.length < remove.length &&
        l.length < insert.length
      ) {
        lead = l;
        remove = remove.slice(l.length);
        insert = insert.slice(l.length);
      }
      const t = remove.match(/\s+$/)?.[0] ?? "";
      if (
        t &&
        insert.endsWith(t) &&
        t.length < remove.length &&
        t.length < insert.length
      ) {
        trail = t;
        remove = remove.slice(0, -t.length);
        insert = insert.slice(0, -t.length);
      }
    }
    pushEqual(lead);
    out.push({ type: "change", remove, insert });
    pushEqual(trail);
  }
  return out;
}

/* ---------- Component ---------- */

const str = (v: unknown) =>
  typeof v === "string" ? v : v == null ? "" : String(v);
/** Pre-computed segments from outside: skips holes and unknown types, and changes that change nothing */
const clean = (list: DiffSegment[]): DiffSegment[] =>
  (Array.isArray(list) ? list : []).flatMap((s): DiffSegment[] => {
    if (s?.type === "equal")
      return str(s.text) ? [{ type: "equal", text: str(s.text) }] : [];
    if (s?.type === "change") {
      const remove = str(s.remove);
      const insert = str(s.insert);
      return remove || insert ? [{ type: "change", remove, insert }] : [];
    }
    return [];
  });
const count = (n: number) => n.toLocaleString("en-US");

type Decisions = Partial<Record<number, InlineDiffVerdict>>;

const plural = (n: number) =>
  `${n.toLocaleString("en-US")} suggestion${n === 1 ? "" : "s"}`;
const quote = (s: string) => `“${s.trim()}”`;
const describe = (c: { remove: string; insert: string }) =>
  c.remove.trim() && c.insert.trim()
    ? `replace ${quote(c.remove)} with ${quote(c.insert)}`
    : c.insert.trim()
      ? `add ${quote(c.insert)}`
      : `remove ${quote(c.remove)}`;

export function InlineDiff({
  original = "",
  suggested = "",
  changes,
  onChange,
  onDone,
  label,
  ref,
  className,
}: InlineDiffProps) {
  const segments = useMemo(
    () =>
      changes
        ? clean(changes)
        : diffWords(String(original ?? ""), String(suggested ?? "")),
    [changes, original, suggested],
  );
  const signature = useMemo(() => JSON.stringify(segments), [segments]);

  // Index of every change among the segments
  const list = useMemo(() => {
    const out: { seg: number; remove: string; insert: string }[] = [];
    segments.forEach((s, i) => {
      if (s.type === "change")
        out.push({ seg: i, remove: s.remove, insert: s.insert });
    });
    return out;
  }, [segments]);

  const [decisions, setDecisions] = useState<Decisions>({});
  const [history, setHistory] = useState<number[][]>([]);
  const [hover, setHover] = useState<number | null>(null);
  const [focused, setFocused] = useState<number | null>(null);
  const [highlighted, setHighlighted] = useState<number | null>(null);
  const [message, setMessage] = useState("");

  // New text in → start over
  const [seen, setSeen] = useState(signature);
  if (seen !== signature) {
    setSeen(signature);
    setDecisions({});
    setHistory([]);
    setHover(null);
    setFocused(null);
    setHighlighted(null);
  }

  const rootRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const popRef = useRef<HTMLDivElement>(null);
  const undoRef = useRef<HTMLButtonElement>(null);
  const items = useRef<(HTMLSpanElement | null)[]>([]);
  const leaveTimer = useRef(0);
  const focusAfter = useRef<"undo" | number | null>(null);
  // The change the arrows step from: the last one shown, navigated to or decided
  const cursor = useRef<number | null>(null);
  const callbacks = useRef({ onChange, onDone });
  callbacks.current = { onChange, onDone };

  const pending = list.map((_, i) => i).filter((i) => !decisions[i]);
  const done = list.length > 0 && pending.length === 0;

  const textFor = (d: Decisions) => {
    let ci = -1;
    return segments
      .map((s) => {
        if (s.type === "equal") return s.text;
        ci++;
        return d[ci] === "accepted" ? s.insert : s.remove;
      })
      .join("");
  };

  // Latest values for the handlers below, so the imperative handle never goes stale
  const live = useRef({ decisions, history, list, pending });
  live.current = { decisions, history, list, pending };

  const decide = (
    ids: number[],
    verdict: InlineDiffVerdict,
    moveFocus = false,
  ) => {
    const { decisions: d, list: l } = live.current;
    const fresh = ids.filter((i) => i >= 0 && i < l.length && !d[i]);
    if (!fresh.length) return;
    const next = { ...d };
    fresh.forEach((i) => (next[i] = verdict));
    const left = l.length - Object.keys(next).length;
    setDecisions(next);
    setHistory((h) => [...h, fresh]);
    setHighlighted(null);
    cursor.current = Math.max(...fresh);
    const word = verdict === "accepted" ? "Accepted" : "Rejected";
    const what =
      fresh.length === 1
        ? describe(l[fresh[0]]).replace(/^./, (c) => c.toUpperCase())
        : `${count(fresh.length)} suggestions`;
    setMessage(
      `${word}: ${what}. ${left ? `${plural(left)} left.` : "All changes reviewed."}`,
    );
    if (moveFocus) {
      const after = Math.max(...fresh);
      const rest = l.map((_, i) => i).filter((i) => !next[i]);
      focusAfter.current = left
        ? (rest.find((i) => i > after) ?? rest[0])
        : "undo";
    }
    const text = textFor(next);
    callbacks.current.onChange?.(text);
    if (!left) callbacks.current.onDone?.(text);
  };

  const undo = (moveFocus = false) => {
    const { decisions: d, history: h } = live.current;
    const last = h[h.length - 1];
    if (!last) return;
    const next = { ...d };
    last.forEach((i) => delete next[i]);
    setDecisions(next);
    setHistory(h.slice(0, -1));
    setMessage(
      `Undone. ${plural(live.current.list.length - Object.keys(next).length)} left.`,
    );
    if (moveFocus) focusAfter.current = Math.min(...last);
    callbacks.current.onChange?.(textFor(next));
  };

  const reset = () => {
    setDecisions({});
    setHistory([]);
    setHighlighted(null);
    setHover(null);
    setMessage("");
    cursor.current = null;
  };

  useImperativeHandle(ref, () => ({
    highlight: (i) => setHighlighted(i),
    accept: (i) => decide([i], "accepted"),
    reject: (i) => decide([i], "rejected"),
    acceptAll: () => decide(live.current.pending, "accepted"),
    rejectAll: () => decide(live.current.pending, "rejected"),
    undo: () => undo(),
    reset,
  }));

  // Move focus once the DOM shows the new state (the Undo layer is inert until then)
  useLayoutEffect(() => {
    const target = focusAfter.current;
    if (target === null) return;
    focusAfter.current = null;
    if (target === "undo") undoRef.current?.focus();
    else items.current[target]?.focus();
  }, [decisions]);

  // Which change shows the popover: the hovered one, else the focused one, else a highlighted one
  const candidates = [hover, focused, highlighted];
  const active =
    candidates.find(
      (i): i is number => i !== null && !decisions[i] && i < list.length,
    ) ?? null;
  useEffect(() => {
    if (active !== null) cursor.current = active;
  }, [active]);

  /*
   * Prev / next: highlight the neighbouring pending change (wrapping around) and show its popover.
   * The arrows never take focus (see onMouseDown below), so the highlight can't be lost to a blur;
   * if a change has keyboard focus, focus follows. The page only scrolls when the change is out of view.
   */
  const step = (dir: 1 | -1) => {
    const p = live.current.pending;
    if (!p.length) return;
    const from = active ?? cursor.current ?? (dir === 1 ? -1 : Infinity);
    const target =
      dir === 1
        ? (p.find((i) => i > from) ?? p[0])
        : ([...p].reverse().find((i) => i < from) ?? p[p.length - 1]);
    window.clearTimeout(leaveTimer.current);
    setHover(null);
    setHighlighted(target);
    cursor.current = target;
    const el = items.current[target];
    if (!el) return;
    if (focused !== null) el.focus({ preventScroll: true });
    const still = window.matchMedia?.(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    el.scrollIntoView({
      block: "nearest",
      inline: "nearest",
      behavior: still ? "auto" : "smooth",
    });
  };

  // A highlight from the arrows stays until a decision, Esc or a click outside the component
  useEffect(() => {
    if (highlighted === null) return;
    const outside = (e: PointerEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setHighlighted(null);
    };
    document.addEventListener("pointerdown", outside);
    return () => document.removeEventListener("pointerdown", outside);
  }, [highlighted]);

  // Place the popover above the first line of the active change, kept inside the card
  const [pos, setPos] = useState<{
    left: number;
    top: number;
    snap: boolean;
    below: boolean;
  } | null>(null);
  const wasOpen = useRef(false);
  useLayoutEffect(() => {
    const place = () => {
      const root = cardRef.current;
      const pop = popRef.current;
      const el = active === null ? null : items.current[active];
      if (!root || !pop || !el) return;
      const r = root.getBoundingClientRect();
      // Works inside CSS zoom or transforms: convert viewport pixels back to the card's own pixels
      const scale = r.width ? root.offsetWidth / r.width : 1;
      const first = el.getClientRects()[0] ?? el.getBoundingClientRect();
      const pw = pop.offsetWidth;
      const ph = pop.offsetHeight;
      const center = (first.left - r.left + first.width / 2) * scale;
      const left = Math.min(
        Math.max(center - pw / 2, 8),
        Math.max(8, root.offsetWidth - pw - 8),
      );
      // Above the line – or below it when the change sits on the card's first line and there's no room above
      const above = (first.top - r.top) * scale - ph - 6;
      const below = above < 4;
      const top = below ? (first.bottom - r.top) * scale + 6 : above;
      setPos({ left, top, snap: !wasOpen.current, below });
      wasOpen.current = true;
    };
    if (active === null) {
      wasOpen.current = false;
      return;
    }
    place();
    const ro = new ResizeObserver(place);
    if (cardRef.current) ro.observe(cardRef.current);
    return () => ro.disconnect();
  }, [active, decisions]);

  const enter = (i: number) => {
    window.clearTimeout(leaveTimer.current);
    setHover(i);
  };
  const leave = () => {
    window.clearTimeout(leaveTimer.current);
    leaveTimer.current = window.setTimeout(() => setHover(null), 140);
  };
  useEffect(() => () => window.clearTimeout(leaveTimer.current), []);

  const onKey = (e: KeyboardEvent, i: number) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "y" || k === "enter") {
      e.preventDefault();
      decide([i], "accepted", true);
    } else if (k === "n" || k === "backspace" || k === "delete") {
      e.preventDefault();
      decide([i], "rejected", true);
    } else if (k === "escape") {
      (e.currentTarget as HTMLElement).blur();
    }
  };

  const total = list.length;
  let ci = -1;

  return (
    <div
      ref={rootRef}
      className={["ui-dfc", className].filter(Boolean).join(" ")}
      data-done={done || undefined}
      onKeyDown={(e) => {
        if (e.key === "Escape" && highlighted !== null) setHighlighted(null);
      }}
    >
      {/* React 19 hoists this into <head> once, no matter how many diffs render */}
      <style href="ui-inline-diff-clean" precedence="default">
        {css}
      </style>

      <div ref={cardRef} className="ui-dfc__card">
        <div className="ui-dfc__body">
          {/* The optional label opens the card */}
          {label && (
            <div className="ui-dfc__label">
              <svg viewBox="0 0 16 16" aria-hidden="true">
                <path d="M8 1.75c.3 2.9 1.35 3.95 4.25 4.25-2.9.3-3.95 1.35-4.25 4.25C7.7 7.35 6.65 6.3 3.75 6 6.65 5.7 7.7 4.65 8 1.75Z" />
                <path d="M12.5 10.25c.15 1.25.6 1.7 1.75 1.75-1.15.15-1.6.6-1.75 1.75-.15-1.15-.6-1.6-1.75-1.75 1.15-.05 1.6-.5 1.75-1.75Z" />
              </svg>
              <span dir="auto" title={label}>
                {label}
              </span>
            </div>
          )}
          <p className="ui-dfc__text" dir="auto">
            {segments.map((s, i) => {
              if (s.type === "equal") return <span key={i}>{s.text}</span>;
              ci++;
              const idx = ci;
              const state: InlineDiffVerdict | "pending" =
                decisions[idx] ?? "pending";
              const open = state === "pending";
              return (
                <span
                  key={i}
                  ref={(el) => {
                    items.current[idx] = el;
                  }}
                  className="ui-dfc__chg"
                  data-state={state}
                  data-active={active === idx || undefined}
                  tabIndex={open ? 0 : undefined}
                  role={open ? "button" : undefined}
                  aria-label={
                    open
                      ? `Suggestion ${count(idx + 1)} of ${count(total)}: ${describe(s)}. Press Y to accept, N to reject`
                      : undefined
                  }
                  onKeyDown={open ? (e) => onKey(e, idx) : undefined}
                  onFocus={() => setFocused(idx)}
                  onBlur={() => setFocused((f) => (f === idx ? null : f))}
                  onMouseEnter={() => enter(idx)}
                  onMouseLeave={leave}
                >
                  {s.remove && (
                    <del
                      className="ui-dfc__del"
                      aria-hidden={state === "accepted" || undefined}
                    >
                      {s.remove}
                    </del>
                  )}
                  {s.insert && (
                    <ins
                      className="ui-dfc__ins"
                      aria-hidden={state === "rejected" || undefined}
                    >
                      {s.insert}
                    </ins>
                  )}
                </span>
              );
            })}
          </p>
        </div>

        {/* The toolbar is the last row of the card, under a hairline */}
        <div className="ui-dfc__foot">
          <div className="ui-dfc__bar">
            <div
              className="ui-dfc__layer"
              data-show={!done || undefined}
              inert={done}
              aria-hidden={done || undefined}
            >
              <span className="ui-dfc__count">
                <span className="ui-dfc__num">{count(pending.length)}</span>
                <span className="ui-dfc__word">
                  {pending.length === 1 ? "suggestion" : "suggestions"}
                </span>
              </span>
              <span className="ui-dfc__actions">
                {/* Not focus-stealing on click: a focused change keeps focus, so its highlight can't blur away */}
                <span
                  className="ui-dfc__nav"
                  onMouseDown={(e) => e.preventDefault()}
                >
                  <button
                    type="button"
                    className="ui-dfc__icon"
                    onClick={() => step(-1)}
                    disabled={!pending.length}
                    aria-label="Previous suggestion"
                  >
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path d="M10 12.5 5.5 8 10 3.5" />
                    </svg>
                  </button>
                  <button
                    type="button"
                    className="ui-dfc__icon"
                    onClick={() => step(1)}
                    disabled={!pending.length}
                    aria-label="Next suggestion"
                  >
                    <svg viewBox="0 0 16 16" aria-hidden="true">
                      <path d="M6 3.5 10.5 8 6 12.5" />
                    </svg>
                  </button>
                </span>
                <button
                  type="button"
                  className="ui-dfc__btn"
                  onClick={() => decide(pending, "rejected")}
                  disabled={!pending.length}
                >
                  Reject all
                </button>
                <button
                  type="button"
                  className="ui-dfc__btn ui-dfc__btn--primary"
                  onClick={() => decide(pending, "accepted")}
                  disabled={!pending.length}
                >
                  Accept all
                </button>
              </span>
            </div>

            <div
              className="ui-dfc__layer"
              data-show={done || undefined}
              inert={!done}
              aria-hidden={!done || undefined}
            >
              <span className="ui-dfc__count ui-dfc__count--done">
                <svg
                  className="ui-dfc__ok"
                  viewBox="0 0 16 16"
                  aria-hidden="true"
                >
                  <path d="M3.75 8.4 6.6 11.1 12.25 5" pathLength={1} />
                </svg>
                All changes reviewed
              </span>
              <button
                ref={undoRef}
                type="button"
                className="ui-dfc__btn"
                onClick={() => undo(true)}
                disabled={!history.length}
              >
                <svg viewBox="0 0 16 16" aria-hidden="true">
                  <path d="M5.5 4 2.75 6.75 5.5 9.5" />
                  <path d="M3 6.75h6.5a3.75 3.75 0 0 1 0 7.5H7.5" />
                </svg>
                Undo
              </button>
            </div>
          </div>
        </div>

        <div
          ref={popRef}
          className="ui-dfc__pop"
          data-open={active !== null || undefined}
          data-snap={pos?.snap || undefined}
          data-below={pos?.below || undefined}
          style={pos ? { left: pos.left, top: pos.top } : undefined}
          role="group"
          aria-label="Review suggestion"
          aria-hidden={active === null || undefined}
          onMouseEnter={() => active !== null && enter(active)}
          onMouseLeave={leave}
          // Keep focus on the change while clicking the popover
          onMouseDown={(e) => e.preventDefault()}
        >
          <button
            type="button"
            tabIndex={-1}
            className="ui-dfc__pbtn ui-dfc__pbtn--primary"
            aria-label="Accept suggestion"
            onClick={() =>
              active !== null &&
              decide([active], "accepted", focused === active)
            }
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M3.75 8.4 6.6 11.1 12.25 5" />
            </svg>
            Accept
            <kbd>Y</kbd>
          </button>
          <button
            type="button"
            tabIndex={-1}
            className="ui-dfc__pbtn"
            aria-label="Reject suggestion"
            onClick={() =>
              active !== null &&
              decide([active], "rejected", focused === active)
            }
          >
            <svg viewBox="0 0 16 16" aria-hidden="true">
              <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />
            </svg>
            Reject
            <kbd>N</kbd>
          </button>
        </div>
      </div>

      <span className="ui-dfc__sr" role="status" aria-live="polite">
        {message}
      </span>
    </div>
  );
}

/* ---------- Styles ---------- */

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-dfc */
.ui-dfc {
  --ui-dfc-card: var(--aui-surface, #ffffff);
  --ui-dfc-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-dfc-strong: var(--aui-ink, #000000);
  --ui-dfc-text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-dfc-muted: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-dfc-faint: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --ui-dfc-icon: var(--aui-muted, rgba(0, 0, 0, 0.447));
  --ui-dfc-line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --ui-dfc-fill: var(--aui-subtle, rgba(0, 0, 0, 0.06));
  --ui-dfc-wash: var(--aui-subtle, rgba(0, 0, 0, 0.03));
  --ui-dfc-accent: rgba(0, 96, 226, 0.765);
  /* Edits are tinted fills only – no strokes */
  --ui-dfc-add: color-mix(in srgb, var(--aui-green, #5fc88a) 10%, transparent);
  --ui-dfc-add-strong: color-mix(in srgb, var(--aui-green, #5fc88a) 16%, transparent);
  --ui-dfc-del: color-mix(in srgb, var(--aui-red, #f04438) 8%, transparent);
  --ui-dfc-del-strong: color-mix(in srgb, var(--aui-red, #f04438) 14%, transparent);
  --ui-dfc-del-ink: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-dfc-strike: color-mix(in srgb, var(--aui-red, #f04438) 60%, transparent);
  --ui-dfc-ok: var(--aui-success, #1e8a4f);
  --ui-dfc-primary: var(--aui-primary, rgba(0, 0, 0, 0.88));
  --ui-dfc-primary-ink: var(--aui-on-primary, #fcfcfc);
  --ui-dfc-primary-kbd: rgba(255, 255, 255, 0.16);
  --ui-dfc-primary-kbd-ink: rgba(255, 255, 255, 0.66);
  --ui-dfc-secondary-hover: #fafafa;
  /* Strokes are shadows: a 0.5px hairline plus a soft lift */
  --ui-dfc-shadow: 0 0 0 0.5px var(--aui-border, rgba(0, 0, 0, 0.06)), 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --ui-dfc-shadow-pop: var(--aui-shadow-popover, 0 0 0 0.5px rgba(0, 0, 0, 0.08), 0 12px 28px -10px rgba(0, 0, 0, 0.16), 0 1px 2px 0 rgba(0, 0, 0, 0.05));
  --ui-dfc-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  --ui-dfc-ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  position: relative;
  box-sizing: border-box;
  display: grid;
  width: 100%;
  min-width: 0;
  color: var(--ui-dfc-ink);
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  line-height: 20px;
  font-weight: 400;
  -webkit-font-smoothing: antialiased;
  text-align: left;
}
.ui-dfc *, .ui-dfc *::before, .ui-dfc *::after { box-sizing: border-box; }
:where(.ui-dfc) button {
  appearance: none;
  margin: 0;
  padding: 0;
  border: 0;
  background: none;
  font: inherit;
  color: inherit;
  cursor: pointer;
  -webkit-tap-highlight-color: transparent;
}
:where(.ui-dfc) svg { display: block; flex: none; }

/* One white card – the edge is a shadow, not a border. The popover is positioned relative to it */
.ui-dfc__card {
  position: relative;
  min-width: 0;
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--ui-dfc-card);
  box-shadow: var(--aui-shadow-card, var(--ui-dfc-shadow));
  container: ui-dfc-card / inline-size;
}

/* The optional label opens the card */
.ui-dfc__label {
  display: flex;
  align-items: center;
  gap: 6px;
  min-width: 0;
  margin: 0 0 8px;
  color: var(--ui-dfc-text);
  font-size: 12px;
  line-height: 16px;
}
.ui-dfc__label span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ui-dfc__label svg { width: 16px; height: 16px; fill: var(--ui-dfc-accent); }

/* The prose */
.ui-dfc__body { padding: 16px 16px 18px; }
.ui-dfc__text {
  margin: 0;
  color: var(--ui-dfc-ink);
  font-size: 13px;
  line-height: 22px;
  text-align: start;
  white-space: pre-line; /* line breaks stay, doubled spaces around a settled change collapse */
  overflow-wrap: anywhere;
}

.ui-dfc__chg { border-radius: 4px; cursor: pointer; scroll-margin: 64px 0 72px; }
.ui-dfc__chg:not([data-state="pending"]) { cursor: auto; }
.ui-dfc__chg:focus { outline: none; }
.ui-dfc__chg:focus-visible { outline: 2px solid var(--ui-dfc-focus); outline-offset: 2px; }

.ui-dfc__del,
.ui-dfc__ins {
  padding: 1px 2px;
  border-radius: 4px;
  -webkit-box-decoration-break: clone;
  box-decoration-break: clone;
  font-size: 1em;
  transition:
    background-color 360ms ease,
    color 360ms ease,
    text-decoration-color 300ms ease,
    font-size 460ms var(--ui-dfc-ease),
    padding 460ms var(--ui-dfc-ease),
    margin 460ms var(--ui-dfc-ease),
    opacity 300ms ease;
}
/* Removed: a red tint, the words grayed and struck */
.ui-dfc__del {
  background: var(--ui-dfc-del);
  color: var(--ui-dfc-del-ink);
  text-decoration: line-through;
  text-decoration-color: var(--ui-dfc-strike);
  text-decoration-thickness: 1px;
}
/* Added: a green tint, the words in full ink */
.ui-dfc__ins {
  background: var(--ui-dfc-add);
  color: var(--ui-dfc-ink);
  text-decoration: none;
}
.ui-dfc__del + .ui-dfc__ins { margin-left: 3px; }
.ui-dfc__chg[data-active] .ui-dfc__del { background: var(--ui-dfc-del-strong); }
.ui-dfc__chg[data-active] .ui-dfc__ins { background: var(--ui-dfc-add-strong); }

/* Accepted: the old words shrink away, the new ones settle into the text */
.ui-dfc__chg[data-state="accepted"] .ui-dfc__del,
.ui-dfc__chg[data-state="rejected"] .ui-dfc__ins {
  font-size: 0;
  padding: 0;
  margin: 0;
  opacity: 0;
}
.ui-dfc__chg[data-state="accepted"] .ui-dfc__ins,
.ui-dfc__chg[data-state="rejected"] .ui-dfc__del {
  padding: 0;
  margin: 0;
  background: transparent;
  color: inherit;
  text-decoration-color: transparent;
}

/* The toolbar: the last row of the card. Its two states share one cell and cross-fade */
.ui-dfc__foot {
  display: grid;
  align-items: center;
  min-height: 44px;
  padding: 8px 8px 8px 16px;
  border-top: 0.5px solid var(--ui-dfc-line);
}
.ui-dfc__bar { display: grid; }
.ui-dfc__layer {
  grid-area: 1 / 1;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
  opacity: 0;
  visibility: hidden;
  transform: translateY(4px);
  filter: blur(2px);
  transition: opacity 260ms ease, transform 420ms var(--ui-dfc-ease), filter 260ms ease, visibility 0s 420ms;
}
.ui-dfc__layer[data-show] {
  opacity: 1;
  visibility: visible;
  transform: none;
  filter: none;
  transition: opacity 320ms ease 80ms, transform 460ms var(--ui-dfc-ease) 80ms, filter 320ms ease 80ms, visibility 0s;
}
/* Narrow cards (e.g. a side panel): the count keeps only its number, so the toolbar stays on one line */
@container ui-dfc-card (max-width: 400px) {
  .ui-dfc__word { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
}
.ui-dfc__count {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  color: var(--ui-dfc-text);
  font-size: 13px;
  line-height: 20px;
  white-space: nowrap;
}
.ui-dfc__count--done { gap: 6px; color: var(--ui-dfc-ink); font-weight: 500; }
.ui-dfc__num {
  display: inline-grid;
  place-items: center;
  min-width: 20px;
  height: 20px;
  padding: 0 6px;
  border-radius: var(--aui-radius-sm, 5px);
  background: var(--ui-dfc-fill);
  color: var(--ui-dfc-ink);
  font-size: 12px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
}
.ui-dfc__ok {
  width: 16px;
  height: 16px;
  fill: none;
  stroke: var(--ui-dfc-ok);
  stroke-width: 1.5;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 1;
  stroke-dashoffset: 1;
}
.ui-dfc[data-done] .ui-dfc__ok { animation: ui-dfc-draw 420ms var(--ui-dfc-ease) 220ms forwards; }
.ui-dfc__actions { display: inline-flex; align-items: center; gap: 8px; margin-left: auto; }
.ui-dfc__nav { display: inline-flex; gap: 2px; }

/* Buttons: 28px, radius 6 – secondary is white with the card shadow, primary a dark chip */
.ui-dfc__btn,
.ui-dfc__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  height: 28px;
  border-radius: calc(var(--aui-radius, 7px) - 1px);
  font-size: 13px;
  font-weight: 500;
  line-height: 20px;
  white-space: nowrap;
  transition: background-color 160ms ease, color 160ms ease, opacity 160ms ease, transform 160ms var(--ui-dfc-ease);
}
.ui-dfc__btn { padding: 0 10px; background: var(--ui-dfc-card); box-shadow: var(--ui-dfc-shadow); color: var(--ui-dfc-strong); }
.ui-dfc__btn:has(svg) { padding: 0 10px 0 8px; }
.ui-dfc__icon { width: 28px; color: var(--ui-dfc-icon); }
.ui-dfc__btn svg,
.ui-dfc__icon svg,
.ui-dfc__pbtn svg {
  width: 16px;
  height: 16px;
  fill: none;
  stroke: currentColor;
  stroke-width: 1.1;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.ui-dfc__btn svg { color: var(--ui-dfc-icon); }
.ui-dfc__btn:hover:not(:disabled) { background: var(--ui-dfc-secondary-hover); }
.ui-dfc__icon:hover:not(:disabled) { background: var(--ui-dfc-wash); color: var(--ui-dfc-ink); }
.ui-dfc__btn--primary { background: var(--ui-dfc-primary); color: var(--ui-dfc-primary-ink); box-shadow: none; }
.ui-dfc__btn--primary:hover:not(:disabled) { background: var(--ui-dfc-primary); opacity: 0.9; }
.ui-dfc__btn:active:not(:disabled),
.ui-dfc__icon:active:not(:disabled) { transform: scale(0.97); }
.ui-dfc__btn:disabled,
.ui-dfc__icon:disabled { opacity: 0.4; cursor: default; }
.ui-dfc__btn:focus-visible,
.ui-dfc__icon:focus-visible { outline: 2px solid var(--ui-dfc-focus); outline-offset: 2px; }

/* Popover: white, radius 10 – Accept is the dark chip, Reject the white secondary button */
.ui-dfc__pop {
  position: absolute;
  top: 0;
  left: 0;
  z-index: 2;
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px;
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--aui-overlay, var(--ui-dfc-card));
  box-shadow: var(--ui-dfc-shadow-pop);
  opacity: 0;
  visibility: hidden;
  transform: translateY(4px) scale(0.96);
  transform-origin: 50% 100%;
  pointer-events: none;
  /* Moving from one change to the next is a short, crisp step – not a long glide */
  transition:
    opacity 160ms ease,
    transform 240ms var(--ui-dfc-ease),
    left 90ms ease-out,
    top 90ms ease-out,
    visibility 0s 240ms;
}
.ui-dfc__pop[data-open] {
  opacity: 1;
  visibility: visible;
  transform: none;
  pointer-events: auto;
  transition:
    opacity 160ms ease,
    transform 280ms var(--ui-dfc-ease),
    left 90ms ease-out,
    top 90ms ease-out,
    visibility 0s;
}
.ui-dfc__pop[data-below] { transform-origin: 50% 0; }
.ui-dfc__pop[data-below]:not([data-open]) { transform: translateY(-4px) scale(0.96); }
.ui-dfc__pop[data-snap] { transition-property: opacity, transform, visibility; }
.ui-dfc__pbtn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  height: 28px;
  padding: 0 6px 0 8px;
  border-radius: calc(var(--aui-radius, 7px) - 1px);
  background: var(--ui-dfc-card);
  box-shadow: var(--ui-dfc-shadow);
  color: var(--ui-dfc-strong);
  font-size: 13px;
  font-weight: 500;
  line-height: 20px;
  white-space: nowrap;
  transition: background-color 140ms ease, opacity 140ms ease;
}
.ui-dfc__pbtn svg { color: var(--ui-dfc-icon); }
.ui-dfc__pbtn:hover { background: var(--ui-dfc-secondary-hover); }
.ui-dfc__pbtn--primary { background: var(--ui-dfc-primary); box-shadow: none; color: var(--ui-dfc-primary-ink); }
.ui-dfc__pbtn--primary svg { color: currentColor; }
.ui-dfc__pbtn--primary:hover { background: var(--ui-dfc-primary); opacity: 0.9; }
.ui-dfc__pbtn kbd {
  display: inline-grid;
  place-items: center;
  min-width: 16px;
  height: 16px;
  padding: 0 4px;
  border-radius: calc(var(--aui-radius-sm, 5px) - 1px);
  background: var(--ui-dfc-fill);
  color: var(--ui-dfc-muted);
  font: inherit;
  font-size: 10px;
  font-weight: 500;
  line-height: 16px;
}
.ui-dfc__pbtn--primary kbd { background: var(--ui-dfc-primary-kbd); color: var(--ui-dfc-primary-kbd-ink); }

.ui-dfc__sr {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}

@keyframes ui-dfc-draw { to { stroke-dashoffset: 0; } }

/* Narrow cards: the arrows step aside (Tab still moves between changes) */
@container ui-dfc-card (max-width: 360px) {
  .ui-dfc__nav { display: none; }
  .ui-dfc__body { padding: 14px 14px 16px; }
  .ui-dfc__foot { padding-left: 14px; }
}
/* Touch screens: taller targets, no key hints */
@media (hover: none) and (pointer: coarse) {
  .ui-dfc__pbtn kbd { display: none; }
  .ui-dfc__pbtn { height: 34px; padding: 0 12px 0 10px; }
  .ui-dfc__btn { height: 34px; }
  .ui-dfc__icon { width: 34px; height: 34px; }
  .ui-dfc__foot { min-height: 52px; }
}

@media (prefers-reduced-motion: reduce) {
  .ui-dfc *, .ui-dfc *::before, .ui-dfc *::after {
    animation-duration: 1ms !important;
    animation-delay: 0ms !important;
    transition-duration: 1ms !important;
    transition-delay: 0ms !important;
  }
  .ui-dfc__layer { filter: none; transform: none; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-dfc {
  --ui-dfc-card: var(--aui-surface, #1c1c1e);
  --ui-dfc-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-dfc-strong: var(--aui-ink, #ffffff);
  --ui-dfc-text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-dfc-muted: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-dfc-faint: var(--aui-placeholder, rgba(255, 255, 255, 0.3));
  --ui-dfc-icon: var(--aui-muted, rgba(255, 255, 255, 0.5));
  --ui-dfc-line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --ui-dfc-fill: var(--aui-subtle, rgba(255, 255, 255, 0.08));
  --ui-dfc-wash: var(--aui-subtle, rgba(255, 255, 255, 0.04));
  --ui-dfc-accent: var(--aui-blue, #4fb0fe);
  --ui-dfc-add: color-mix(in srgb, var(--aui-green, #7dcf9f) 14%, transparent);
  --ui-dfc-add-strong: color-mix(in srgb, var(--aui-green, #7dcf9f) 22%, transparent);
  --ui-dfc-del: color-mix(in srgb, var(--aui-red, #ff6b5f) 14%, transparent);
  --ui-dfc-del-strong: color-mix(in srgb, var(--aui-red, #ff6b5f) 22%, transparent);
  --ui-dfc-del-ink: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-dfc-strike: color-mix(in srgb, var(--aui-red, #ff6b5f) 70%, transparent);
  --ui-dfc-ok: var(--aui-green, #5fc88a);
  --ui-dfc-primary: var(--aui-primary, rgba(255, 255, 255, 0.9));
  --ui-dfc-primary-ink: var(--aui-on-primary, #1c1c1e);
  --ui-dfc-primary-kbd: rgba(0, 0, 0, 0.08);
  --ui-dfc-primary-kbd-ink: rgba(28, 28, 30, 0.6);
  --ui-dfc-secondary-hover: #232326;
  --ui-dfc-shadow: 0 0 0 0.5px var(--aui-border, rgba(255, 255, 255, 0.1)), 0 1px 2px 0 rgba(0, 0, 0, 0.4);
  --ui-dfc-shadow-pop: var(--aui-shadow-popover, 0 0 0 0.5px rgba(255, 255, 255, 0.12), 0 12px 28px -10px rgba(0, 0, 0, 0.6));
  --ui-dfc-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
}
:where(.dark, [data-theme="dark"]) .ui-dfc__pop { --ui-dfc-card: #232326; --ui-dfc-secondary-hover: #2a2a2d; }
`;
