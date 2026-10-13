import { useRef, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { useMountEffect } from "@/hooks/useMountEffect";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { readDraftCache, writeDraftCache } from "./draftCache";
import type { DraftTarget } from "./draftTargets";

const MAX_DRAFT_BYTES = 240_000;

interface Args {
  target: DraftTarget | null;
  source: string;
  epoch: number;
  baseRevision: number;
  enabled: boolean;
}

interface Failure {
  epoch: number;
  conflict: boolean;
  message: string;
}

export function useDraftSave(args: Args) {
  const queryClient = useQueryClient();
  const live = useRef(args);
  live.current = args;
  const written = useRef({ epoch: -1, revision: 0, serialized: "" });
  const conflictEpoch = useRef(-1);
  const latest = useRef<Record<string, unknown> | null>(null);
  const busy = useRef(false);
  const mounted = useRef(true);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const [failure, setFailure] = useState<Failure | null>(null);
  const [saving, setSaving] = useState(false);

  useMountEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      clearTimeout(timer.current);
    };
  });

  function savedFor(epoch: number) {
    return written.current.epoch === epoch
      ? written.current
      : { epoch, revision: live.current.baseRevision, serialized: "" };
  }

  async function flush() {
    const { target, source, epoch, enabled } = live.current;
    if (!target || !enabled || busy.current || !latest.current) return;
    if (conflictEpoch.current === epoch) return;
    const state = latest.current;
    const serialized = JSON.stringify(state);
    const saved = savedFor(epoch);
    if (serialized === saved.serialized) return;
    if (new TextEncoder().encode(serialized).length > MAX_DRAFT_BYTES) {
      setFailure({
        epoch,
        conflict: false,
        message:
          "This draft is too large to sync. Download your data before leaving.",
      });
      return;
    }
    busy.current = true;
    setSaving(true);
    setFailure(null);
    try {
      const revision = await target.write(state, saved.revision);
      written.current = { epoch, revision, serialized };
      const current = latest.current ?? state;
      if (mounted.current)
        writeDraftCache(target.cacheKey, {
          source,
          state: current,
          revision,
          dirty: JSON.stringify(current) !== serialized,
        });
      void queryClient.invalidateQueries({ queryKey: target.queryKey });
    } catch (cause) {
      const conflict = cause instanceof ApiError && cause.status === 409;
      if (conflict) conflictEpoch.current = epoch;
      if (mounted.current)
        setFailure({
          epoch,
          conflict,
          message: conflict
            ? "A newer draft exists. Your edits are still here; reload the saved draft before editing further."
            : "Draft sync failed. Your edits are still here. Retry before leaving this device.",
        });
    } finally {
      busy.current = false;
      if (mounted.current) setSaving(false);
    }
    if (
      mounted.current &&
      written.current.serialized === serialized &&
      latest.current &&
      JSON.stringify(latest.current) !== serialized
    )
      timer.current = setTimeout(() => void flush(), 500);
  }

  function save(state: Record<string, unknown>) {
    const { target, source, epoch } = live.current;
    const saved = savedFor(epoch);
    latest.current = state;
    writeDraftCache(target?.cacheKey ?? null, {
      source,
      state,
      revision: saved.revision,
      dirty: JSON.stringify(state) !== saved.serialized,
    });
    clearTimeout(timer.current);
    timer.current = setTimeout(() => void flush(), 800);
  }

  async function flushNow() {
    clearTimeout(timer.current);
    await flush();
  }

  function reset() {
    clearTimeout(timer.current);
    latest.current = null;
  }

  function syncedRevision() {
    return savedFor(live.current.epoch).revision;
  }

  // Edits here that the server does not have yet: a pending or failed write,
  // or a draft restored from this browser's unsynced cache.
  function hasUnsyncedEdits() {
    const { target, source, epoch } = live.current;
    if (busy.current) return true;
    if (latest.current)
      return JSON.stringify(latest.current) !== savedFor(epoch).serialized;
    if (written.current.epoch === epoch) return false;
    const cached = readDraftCache(target?.cacheKey ?? null);
    return !!cached && cached.source === source && cached.dirty !== false;
  }

  const current = failure?.epoch === args.epoch ? failure : null;
  return {
    save,
    retry: flush,
    flushNow,
    reset,
    syncedRevision,
    hasUnsyncedEdits,
    error: current?.message ?? "",
    conflict: current?.conflict ?? false,
    saving,
    latest,
  };
}
