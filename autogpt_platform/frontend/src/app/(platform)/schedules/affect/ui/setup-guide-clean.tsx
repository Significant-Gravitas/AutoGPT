"use client";

import { useState } from "react";

/*
 * Setup Guide – an onboarding panel with a progress bar and collapsible task groups, in a single self-contained file.
 * Tasks unlock one after another; finished tasks and groups are struck through.
 * The styles are injected below, so no extra files are needed.
 * Clean style: one white card with hairline shadows, quiet gray type, a thin blue progress line with
 * a count, flat group rows with a soft wash on the open one, and blue checks.
 */

export type SetupTask = {
  id: string;
  label: string;
  done?: boolean;
};

export type SetupGroup = {
  id: string;
  title: string;
  tasks: SetupTask[];
};

export type SetupGuideProps = {
  title?: string;
  groups: SetupGroup[];
  /** Shows an "Edit" link in the header */
  onEdit?: () => void;
  /** Shows a close button in the header */
  onClose?: () => void;
  /** Called with the ids of all finished tasks */
  onChange?: (done: string[]) => void;
  className?: string;
};

const count = new Intl.NumberFormat("en");

export function SetupGuide({
  title = "Setup guide",
  groups,
  onEdit,
  onClose,
  onChange,
  className,
}: SetupGuideProps) {
  const [done, setDone] = useState(
    () =>
      new Set(
        groups.flatMap((g) => g.tasks.filter((t) => t.done).map((t) => t.id)),
      ),
  );
  const firstOpen = groups.find((g) => g.tasks.some((t) => !done.has(t.id)));
  const [open, setOpen] = useState<string | null>(firstOpen?.id ?? null);
  const [minimized, setMinimized] = useState(false);

  const total = groups.reduce((n, g) => n + g.tasks.length, 0);
  // Counted from the tasks shown, so ids that aren't (or are twice) in the list can't skew it
  const doneCount = groups.reduce(
    (n, g) => n + g.tasks.filter((t) => done.has(t.id)).length,
    0,
  );
  const progress = total ? Math.min(1, doneCount / total) : 0;

  const toggle = (group: SetupGroup, id: string) => {
    const next = new Set(done);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setDone(next);
    onChange?.([...next]);
    // A finished group folds away and the next open one unfolds
    if (group.tasks.every((t) => next.has(t.id))) {
      const following = groups.find((g) =>
        g.tasks.some((t) => !next.has(t.id)),
      );
      window.setTimeout(() => setOpen(following?.id ?? null), 420);
    }
  };

  return (
    <>
      <style href="ui-setup-guide-clean" precedence="default">
        {css}
      </style>
      <section
        className={["ui-sgdc", className].filter(Boolean).join(" ")}
        aria-label={title}
      >
        <header className="ui-sgdc__head">
          <h3 className="ui-sgdc__title" title={title} dir="auto">
            {title}
          </h3>
          <div className="ui-sgdc__actions">
            {onEdit && (
              <button type="button" className="ui-sgdc__edit" onClick={onEdit}>
                Edit
              </button>
            )}
            <button
              type="button"
              className="ui-sgdc__icon-btn"
              aria-label={
                minimized ? "Expand setup guide" : "Minimize setup guide"
              }
              aria-expanded={!minimized}
              onClick={() => setMinimized((m) => !m)}
            >
              {minimized ? <ExpandIcon /> : <MinimizeIcon />}
            </button>
            {onClose && (
              <button
                type="button"
                className="ui-sgdc__icon-btn"
                aria-label="Close setup guide"
                onClick={onClose}
              >
                <CloseIcon />
              </button>
            )}
          </div>
        </header>

        <div className="ui-sgdc__progress">
          <span
            className="ui-sgdc__track"
            role="progressbar"
            aria-label={`${title} progress`}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-valuenow={Math.round(progress * 100)}
          >
            <span
              className="ui-sgdc__fill"
              style={{ transform: `scaleX(${progress})` }}
            />
          </span>
          <span className="ui-sgdc__count" aria-hidden="true">
            {count.format(doneCount)}/{count.format(total)}
          </span>
        </div>

        <div className="ui-sgdc__body" data-hidden={minimized || undefined}>
          <div className="ui-sgdc__groups">
            {groups.map((group, groupIndex) => {
              const isOpen = open === group.id;
              const finished = group.tasks.every((t) => done.has(t.id));
              // The first open task is the current one, everything after it waits
              const current = group.tasks.find((t) => !done.has(t.id))?.id;
              const currentIndex = group.tasks.findIndex(
                (t) => t.id === current,
              );
              const panelId = `ui-sgdc-${group.id}`;
              return (
                <div
                  key={`${groupIndex}-${group.id}`}
                  className="ui-sgdc__group"
                  data-open={isOpen || undefined}
                >
                  <button
                    type="button"
                    className="ui-sgdc__group-head"
                    aria-expanded={isOpen}
                    aria-controls={panelId}
                    onClick={() => setOpen(isOpen ? null : group.id)}
                  >
                    <span
                      className="ui-sgdc__group-title"
                      data-done={finished || undefined}
                      dir="auto"
                    >
                      {group.title}
                    </span>
                    <ChevronIcon />
                  </button>
                  <div className="ui-sgdc__panel" id={panelId} inert={!isOpen}>
                    <ul className="ui-sgdc__tasks">
                      {group.tasks.map((task, index) => {
                        const isDone = done.has(task.id);
                        const locked =
                          !isDone &&
                          currentIndex !== -1 &&
                          index > currentIndex;
                        const state = isDone
                          ? "done"
                          : locked
                            ? "locked"
                            : "current";
                        return (
                          <li key={`${index}-${task.id}`}>
                            <button
                              type="button"
                              className="ui-sgdc__task"
                              data-state={state}
                              disabled={locked}
                              aria-pressed={isDone}
                              onClick={() => toggle(group, task.id)}
                            >
                              <span
                                className="ui-sgdc__mark"
                                aria-hidden="true"
                              >
                                {state === "done" ? (
                                  <CheckIcon />
                                ) : state === "locked" ? (
                                  <LockedIcon />
                                ) : null}
                              </span>
                              <span className="ui-sgdc__task-label" dir="auto">
                                {task.label}
                              </span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </section>
    </>
  );
}

/* ---------- Icons (16px, thin lines) ---------- */

const ChevronIcon = () => (
  <svg
    className="ui-sgdc__chevron"
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.3"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="m4.5 6.25 3.5 3.5 3.5-3.5" />
  </svg>
);

const MinimizeIcon = () => (
  <svg
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M9.5 2.5v4h4M9.5 6.5l4.5-4.5M6.5 13.5v-4h-4M6.5 9.5 2 14" />
  </svg>
);

const ExpandIcon = () => (
  <svg
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.2"
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M10 2h4v4M14 2 9.5 6.5M6 14H2v-4M2 14l4.5-4.5" />
  </svg>
);

const CloseIcon = () => (
  <svg
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.2"
    strokeLinecap="round"
    aria-hidden="true"
  >
    <path d="m4 4 8 8M12 4l-8 8" />
  </svg>
);

const CheckIcon = () => (
  <svg viewBox="0 0 16 16" aria-hidden="true">
    <circle cx="8" cy="8" r="7" fill="currentColor" />
    <path
      d="m5.2 8.2 1.9 1.9 3.7-3.9"
      fill="none"
      stroke="var(--ui-sgdc-on-accent)"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const LockedIcon = () => (
  <svg
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.1"
    aria-hidden="true"
  >
    <circle cx="8" cy="8" r="6.5" strokeDasharray="2 2" />
  </svg>
);

/* ---------- Styles ---------- */

const css = /* css */ `
/* Colors and motion as variables – override them on .ui-sgdc */
.ui-sgdc {
  --ui-sgdc-card: var(--aui-surface, #ffffff);
  --ui-sgdc-strong: var(--aui-ink, #000000);
  --ui-sgdc-ink: var(--aui-ink, rgba(0, 0, 0, 0.88));
  --ui-sgdc-text: var(--aui-muted, rgba(0, 0, 0, 0.61));
  --ui-sgdc-muted: var(--aui-muted, rgba(0, 0, 0, 0.45));
  --ui-sgdc-faint: var(--aui-placeholder, rgba(0, 0, 0, 0.27));
  --ui-sgdc-icon: var(--aui-muted, rgba(0, 0, 0, 0.447));
  --ui-sgdc-line: var(--aui-border, rgba(0, 0, 0, 0.06));
  --ui-sgdc-fill: rgba(0, 0, 0, 0.06);
  --ui-sgdc-ring: var(--aui-border-strong, rgba(0, 0, 0, 0.2));
  --ui-sgdc-wash: var(--aui-subtle, rgba(0, 0, 0, 0.03));
  --ui-sgdc-open: rgba(0, 0, 0, 0.025);
  --ui-sgdc-accent: rgba(0, 96, 226, 0.765);
  --ui-sgdc-on-accent: #ffffff;
  --ui-sgdc-focus: var(--aui-focus, rgba(0, 96, 226, 0.7));
  /* Strokes are shadows: a 0.5px hairline plus a soft lift */
  --ui-sgdc-shadow: var(--aui-shadow-card, 0 0 0 0.5px rgba(0, 0, 0, 0.06), 0 1px 1px -1px rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.05));
  --ui-sgdc-ease: var(--aui-ease, cubic-bezier(0.22, 1, 0.36, 1));
  box-sizing: border-box;
  container-type: inline-size;
  width: 100%;
  max-width: 320px;
  border-radius: var(--aui-radius-lg, 10px);
  background: var(--ui-sgdc-card);
  box-shadow: var(--ui-sgdc-shadow);
  font-family: var(--aui-font, "Inter", "InterVariable", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif);
  font-size: 13px;
  font-weight: 400;
  line-height: 20px;
  color: var(--ui-sgdc-ink);
  -webkit-font-smoothing: antialiased;
}
.ui-sgdc *, .ui-sgdc *::before, .ui-sgdc *::after { box-sizing: border-box; }
:where(.ui-sgdc) button { margin: 0; padding: 0; border: 0; background: none; font: inherit; color: inherit; cursor: pointer; }
:where(.ui-sgdc) svg { display: block; flex: none; }
.ui-sgdc button:focus-visible { outline: 2px solid var(--ui-sgdc-focus); outline-offset: 2px; }

/* ---- Header: title, Edit and the two ghost icon buttons ---- */
.ui-sgdc__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 12px 10px 0 16px;
}
.ui-sgdc__title {
  overflow: hidden;
  min-width: 0;
  margin: 0;
  font-size: 15px;
  font-weight: 500;
  line-height: 24px;
  color: var(--ui-sgdc-strong);
  text-align: start;
  white-space: nowrap;
  text-overflow: ellipsis;
}
.ui-sgdc__actions { display: flex; flex: none; align-items: center; gap: 2px; }
.ui-sgdc__edit {
  height: 28px;
  padding: 0 8px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  font-size: 13px;
  font-weight: 500;
  line-height: 20px;
  color: var(--ui-sgdc-text);
  transition: background-color 100ms ease, color 100ms ease;
}
.ui-sgdc__edit:hover { background: var(--ui-sgdc-wash); color: var(--ui-sgdc-ink); }
.ui-sgdc__icon-btn {
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  color: var(--ui-sgdc-icon);
  transition: background-color 100ms ease, color 100ms ease;
}
.ui-sgdc__icon-btn:hover { background: var(--ui-sgdc-wash); color: var(--ui-sgdc-ink); }
.ui-sgdc__icon-btn svg { width: 16px; height: 16px; }

/* ---- One thin blue line for all tasks, with the count next to it ---- */
.ui-sgdc__progress {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 16px 12px;
}
.ui-sgdc__track {
  display: block;
  flex: 1;
  height: 4px;
  overflow: hidden;
  border-radius: 2px;
  background: var(--ui-sgdc-fill);
}
.ui-sgdc__fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: var(--ui-sgdc-accent);
  transform-origin: left center;
  transition: transform 700ms var(--ui-sgdc-ease);
}
.ui-sgdc__count {
  flex: none;
  font-size: 12px;
  line-height: 16px;
  font-variant-numeric: tabular-nums;
  color: var(--ui-sgdc-muted);
}

/* ---- Minimize folds the whole list away (the hairline above it too) ---- */
.ui-sgdc__body {
  display: grid;
  grid-template-rows: 1fr;
  transition: grid-template-rows 420ms var(--ui-sgdc-ease), opacity 260ms ease;
}
.ui-sgdc__body[data-hidden] { grid-template-rows: 0fr; opacity: 0; }
.ui-sgdc__groups {
  display: grid;
  gap: 2px;
  min-height: 0;
  overflow: hidden;
  padding: 0 4px;
  border-top: 0.5px solid var(--ui-sgdc-line);
}
/* Inner padding as a spacer, so focus rings aren't cut and the fold can still close to 0 */
.ui-sgdc__groups::before, .ui-sgdc__groups::after { content: ""; display: block; height: 2px; }

.ui-sgdc__group {
  border-radius: calc(var(--aui-radius-lg, 10px) - 2px);
  transition: background-color 200ms ease;
}
.ui-sgdc__group[data-open] { background: var(--ui-sgdc-open); }
.ui-sgdc__group-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  width: 100%;
  padding: 8px 8px 8px 12px;
  border-radius: calc(var(--aui-radius-lg, 10px) - 2px);
  text-align: left;
  transition: background-color 100ms ease;
}
.ui-sgdc__group-head:hover { background: var(--ui-sgdc-wash); }
.ui-sgdc__group[data-open] .ui-sgdc__group-head:hover { background: none; }
.ui-sgdc__group-title {
  position: relative;
  min-width: 0;
  font-size: 13px;
  font-weight: 500;
  line-height: 20px;
  color: var(--ui-sgdc-strong);
  text-align: start;
  overflow-wrap: anywhere;
  transition: color 300ms ease;
}
.ui-sgdc__group-title[data-done] { color: var(--ui-sgdc-muted); }
/* Finished: a hairline draws through the title */
.ui-sgdc__group-title::after {
  content: "";
  position: absolute;
  left: 0;
  right: 0;
  top: 52%;
  height: 1px;
  background: currentColor;
  transform: scaleX(0);
  transform-origin: left center;
  transition: transform 420ms var(--ui-sgdc-ease);
}
.ui-sgdc__group-title[data-done]::after { transform: scaleX(1); }
.ui-sgdc__chevron {
  width: 16px;
  height: 16px;
  margin-top: 2px;
  color: var(--ui-sgdc-icon);
  transition: transform 320ms var(--ui-sgdc-ease);
}
.ui-sgdc__group[data-open] .ui-sgdc__chevron { transform: rotate(180deg); }

/* ---- Each group is a dropdown ---- */
.ui-sgdc__panel {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 380ms var(--ui-sgdc-ease);
}
.ui-sgdc__group[data-open] .ui-sgdc__panel { grid-template-rows: 1fr; }
.ui-sgdc__tasks {
  min-height: 0;
  overflow: hidden;
  margin: 0;
  padding: 0 4px;
  list-style: none;
}
.ui-sgdc__tasks li:last-child { padding-bottom: 6px; }
.ui-sgdc__task {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  width: 100%;
  padding: 6px 8px;
  border-radius: calc(var(--aui-radius-sm, 5px) + 1px);
  text-align: left;
  opacity: 0;
  transform: translateY(-4px);
  transition: opacity 260ms ease, transform 320ms var(--ui-sgdc-ease), color 260ms ease, background-color 100ms ease;
}
.ui-sgdc__group[data-open] .ui-sgdc__task { opacity: 1; transform: none; transition-delay: 80ms, 80ms, 0ms, 0ms; }
.ui-sgdc__task:not(:disabled):hover { background: var(--ui-sgdc-wash); }
.ui-sgdc__task[data-state="current"] { color: var(--ui-sgdc-ink); }
.ui-sgdc__task[data-state="locked"] { color: var(--ui-sgdc-muted); cursor: default; }
.ui-sgdc__task[data-state="done"] { color: var(--ui-sgdc-muted); }

/* The mark: an open ring for the current task, a dashed faint ring while locked, a blue check when done */
.ui-sgdc__mark {
  display: grid;
  place-items: center;
  flex: none;
  width: 16px;
  height: 16px;
  margin-top: 2px;
  border-radius: 50%;
  transition: box-shadow 160ms ease;
}
.ui-sgdc__task[data-state="current"] .ui-sgdc__mark { box-shadow: inset 0 0 0 1px var(--ui-sgdc-ring); }
.ui-sgdc__task[data-state="current"]:hover .ui-sgdc__mark { box-shadow: inset 0 0 0 1px var(--ui-sgdc-accent); }
.ui-sgdc__task[data-state="locked"] .ui-sgdc__mark { color: var(--ui-sgdc-faint); }
.ui-sgdc__task[data-state="done"] .ui-sgdc__mark { color: var(--ui-sgdc-accent); }
.ui-sgdc__mark svg { width: 16px; height: 16px; }
.ui-sgdc__task[data-state="done"] .ui-sgdc__mark svg { animation: ui-sgdc-pop 420ms var(--ui-sgdc-ease); }
.ui-sgdc__task-label {
  min-width: 0;
  font-size: 13px;
  line-height: 20px;
  text-align: start;
  white-space: pre-line;
  overflow-wrap: anywhere;
  text-decoration-line: line-through;
  text-decoration-color: transparent;
  text-decoration-thickness: 1px;
  transition: text-decoration-color 260ms ease;
}
.ui-sgdc__task[data-state="done"] .ui-sgdc__task-label { text-decoration-color: currentColor; }

@keyframes ui-sgdc-pop {
  from { transform: scale(0.5); opacity: 0; }
}

/* Narrow: a little less side padding */
@container (max-width: 300px) {
  .ui-sgdc__head { padding-left: 12px; }
  .ui-sgdc__progress { padding-inline: 12px; }
  .ui-sgdc__group-head { padding-left: 10px; }
}

/* Touch: taller targets */
@media (pointer: coarse) {
  .ui-sgdc__icon-btn { width: 36px; height: 36px; }
  .ui-sgdc__edit { height: 36px; }
  .ui-sgdc__group-head { padding-block: 12px; }
  .ui-sgdc__task { padding-block: 9px; }
}

@media (prefers-reduced-motion: reduce) {
  .ui-sgdc__fill, .ui-sgdc__body, .ui-sgdc__panel, .ui-sgdc__task, .ui-sgdc__chevron, .ui-sgdc__group, .ui-sgdc__group-title::after { transition: none; }
  .ui-sgdc__mark svg { animation: none; }
}

/* Dark mode: follows a .dark class or data-theme="dark" on any ancestor, e.g. <html> */
:where(.dark, [data-theme="dark"]) .ui-sgdc {
  --ui-sgdc-card: var(--aui-surface, #1c1c1e);
  --ui-sgdc-strong: var(--aui-ink, #ffffff);
  --ui-sgdc-ink: var(--aui-ink, rgba(255, 255, 255, 0.9));
  --ui-sgdc-text: var(--aui-muted, rgba(255, 255, 255, 0.62));
  --ui-sgdc-muted: var(--aui-muted, rgba(255, 255, 255, 0.46));
  --ui-sgdc-faint: var(--aui-placeholder, rgba(255, 255, 255, 0.3));
  --ui-sgdc-icon: var(--aui-muted, rgba(255, 255, 255, 0.5));
  --ui-sgdc-line: var(--aui-border, rgba(255, 255, 255, 0.08));
  --ui-sgdc-fill: rgba(255, 255, 255, 0.08);
  --ui-sgdc-ring: var(--aui-border-strong, rgba(255, 255, 255, 0.26));
  --ui-sgdc-wash: var(--aui-subtle, rgba(255, 255, 255, 0.04));
  --ui-sgdc-open: rgba(255, 255, 255, 0.03);
  --ui-sgdc-accent: var(--aui-blue, #4fb0fe);
  --ui-sgdc-on-accent: #ffffff;
  --ui-sgdc-focus: var(--aui-focus, rgba(77, 141, 255, 0.8));
  --ui-sgdc-shadow: var(--aui-shadow-card, 0 0 0 0.5px rgba(255, 255, 255, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.4));
}
`;
