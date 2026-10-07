"use client";

import { Button } from "@/components/atoms/Button/Button";
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
      <span className="mx-2 h-5 border-l border-zinc-200" />
      <Button
        type="button"
        variant="primary"
        size="md"
        onClick={onDream}
        disabled={dreamActive}
        title="Run ONLY the dream pass (consolidate → recombine → sanitize) — skips community rebuild and ratification."
      >
        {dreamActive ? jobButtonLabel(dreamStatus, "Dreaming…") : "Dream pass"}
      </Button>
      <Button
        type="button"
        variant="primary"
        size="md"
        onClick={onRatification}
        disabled={ratificationPending}
        title="Run ONLY the ratification supersession sweep — promotes hit tentatives, supersedes unratified ones past their grace period."
      >
        {ratificationPending ? "Ratifying…" : "Ratification"}
      </Button>
      <Button
        type="button"
        variant="primary"
        size="md"
        onClick={onNightly}
        disabled={nightlyActive}
        title="Run the FULL nightly batch — what the 03:00 cron does. Fans out dream pass + ratification sweep (+ future P2/P3/P4/P11 stages) in one pass."
      >
        {nightlyActive
          ? jobButtonLabel(nightlyStatus, "Running nightly…")
          : "Nightly batch"}
      </Button>
      <span className="mx-2 h-5 border-l border-zinc-200" />
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
      <Button
        type="button"
        variant="primary"
        size="md"
        onClick={onRebuild}
        disabled={active}
      >
        {active ? jobButtonLabel(status, "Rebuilding…") : "Rebuild communities"}
      </Button>
      <label className="flex items-center gap-2 text-zinc-700">
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
