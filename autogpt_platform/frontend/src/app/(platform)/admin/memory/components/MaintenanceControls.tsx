"use client";

import type { CommunityRebuildJobStatus } from "@/app/api/__generated__/models/communityRebuildJobStatus";
import type { DreamJobStatus } from "@/app/api/__generated__/models/dreamJobStatus";
import type { NightlyJobStatus } from "@/app/api/__generated__/models/nightlyJobStatus";

type AnyJobStatus =
  | DreamJobStatus
  | NightlyJobStatus
  | CommunityRebuildJobStatus;

type Props = {
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
};

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
      <span className="mx-2 h-5 border-l border-gray-200" />
      <button
        type="button"
        onClick={onDream}
        disabled={dreamActive}
        className="rounded-md bg-purple-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50"
        title="Run ONLY the dream pass (consolidate → recombine → sanitize) — skips community rebuild and ratification."
      >
        {dreamActive ? jobButtonLabel(dreamStatus, "Dreaming…") : "Dream pass"}
      </button>
      <button
        type="button"
        onClick={onRatification}
        disabled={ratificationPending}
        className="rounded-md bg-teal-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-teal-700 disabled:opacity-50"
        title="Run ONLY the ratification supersession sweep — promotes hit tentatives, supersedes unratified ones past their grace period."
      >
        {ratificationPending ? "Ratifying…" : "Ratification"}
      </button>
      <button
        type="button"
        onClick={onNightly}
        disabled={nightlyActive}
        className="rounded-md bg-indigo-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-indigo-800 disabled:opacity-50"
        title="Run the FULL nightly batch — what the 03:00 cron does. Fans out dream pass + ratification sweep (+ future P2/P3/P4/P11 stages) in one pass."
      >
        {nightlyActive
          ? jobButtonLabel(nightlyStatus, "Running nightly…")
          : "Nightly batch"}
      </button>
      <span className="mx-2 h-5 border-l border-gray-200" />
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
        className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {active ? jobButtonLabel(status, "Rebuilding…") : "Rebuild communities"}
      </button>
      <label className="flex items-center gap-2 text-gray-700">
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
