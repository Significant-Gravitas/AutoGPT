"use client";

import type { AnyJobStatus } from "./memoryJobStatus";

interface Props {
  onRebuild: () => void;
  rebuildActive: boolean;
  rebuildStatus: AnyJobStatus | undefined;
  force: boolean;
  setForce: (value: boolean) => void;
  onDream: () => void;
  dreamActive: boolean;
  dreamStatus: AnyJobStatus | undefined;
  onRatification: () => void;
  ratificationPending: boolean;
  onNightly: () => void;
  nightlyActive: boolean;
  nightlyStatus: AnyJobStatus | undefined;
}

export function MaintenanceControls({
  onRebuild,
  rebuildActive,
  rebuildStatus,
  force,
  setForce,
  onDream,
  dreamActive,
  dreamStatus,
  onRatification,
  ratificationPending,
  onNightly,
  nightlyActive,
  nightlyStatus,
}: Props) {
  return (
    <>
      <RebuildControl
        onRebuild={onRebuild}
        active={rebuildActive}
        status={rebuildStatus}
        force={force}
        setForce={setForce}
      />
      <span className="mx-2 h-5 border-l border-border-base" />
      <button
        type="button"
        onClick={onDream}
        disabled={dreamActive}
        className="rounded-md bg-brand-hover px-3 py-1.5 text-sm font-medium text-action-foreground hover:bg-accent-strong disabled:opacity-50"
        title="Run ONLY the dream pass (consolidate → recombine → sanitize) — skips community rebuild and ratification."
      >
        {dreamActive ? jobButtonLabel(dreamStatus, "Dreaming…") : "Dream pass"}
      </button>
      <button
        type="button"
        onClick={onRatification}
        disabled={ratificationPending}
        // oxlint-disable-next-line design-system/no-raw-colors -- teal has no role; a role would change the hue (deferred)
        className="rounded-md bg-teal-600 px-3 py-1.5 text-sm font-medium text-action-foreground hover:bg-teal-700 disabled:opacity-50"
        title="Run ONLY the ratification supersession sweep — promotes hit tentatives, supersedes unratified ones past their grace period."
      >
        {ratificationPending ? "Ratifying…" : "Ratification"}
      </button>
      <button
        type="button"
        onClick={onNightly}
        disabled={nightlyActive}
        // oxlint-disable-next-line design-system/no-raw-colors -- indigo has no role; info or brand would change the hue (deferred)
        className="rounded-md bg-indigo-700 px-3 py-1.5 text-sm font-medium text-action-foreground hover:bg-indigo-800 disabled:opacity-50"
        title="Run the FULL nightly batch — what the 03:00 cron does. Fans out dream pass + ratification sweep (+ future P2/P3/P4/P11 stages) in one pass."
      >
        {nightlyActive
          ? jobButtonLabel(nightlyStatus, "Running nightly…")
          : "Nightly batch"}
      </button>
      <span className="mx-2 h-5 border-l border-border-base" />
    </>
  );
}

type RebuildControlProps = Pick<Props, "onRebuild" | "force" | "setForce"> & {
  active: boolean;
  status: AnyJobStatus | undefined;
};

function RebuildControl({
  onRebuild,
  active,
  status,
  force,
  setForce,
}: RebuildControlProps) {
  return (
    <>
      <button
        type="button"
        onClick={onRebuild}
        disabled={active}
        className="rounded-md bg-info-foreground px-3 py-1.5 text-sm font-medium text-action-foreground hover:bg-info-foreground/90 disabled:opacity-50"
      >
        {active ? jobButtonLabel(status, "Rebuilding…") : "Rebuild communities"}
      </button>
      <label className="flex items-center gap-2 text-foreground-body">
        <input
          type="checkbox"
          checked={force}
          onChange={(event) => setForce(event.target.checked)}
        />
        Force
      </label>
    </>
  );
}

function jobButtonLabel(
  status: AnyJobStatus | undefined,
  fallback: string,
): string {
  if (!status) return fallback;
  if (status.state === "submitted") {
    return status.current_phase
      ? `Batch submitted (${status.current_phase})…`
      : "Batch submitted…";
  }
  if (status.current_phase) {
    return `${capitalize(status.current_phase)}…`;
  }
  return fallback;
}

function capitalize(value: string): string {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
