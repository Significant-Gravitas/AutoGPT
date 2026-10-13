import { useState } from "react";
import { persistedState } from "@/lib/openui/draft-fields";
import { downloadText } from "@/lib/openui/download";
import { readUIDraft, saveUIDraft } from "./helpers";
import { readDraftCache, removeDraftCache } from "./draftCache";
import type { DraftRemote, DraftTarget } from "./draftTargets";
import {
  restoreDraft,
  SUPERSEDED_NOTICE,
  type DraftSnapshot,
} from "./draftRestore";
import { useDraftSave } from "./useDraftSave";

interface Args {
  namespace: string;
  target: DraftTarget | null;
  remote: DraftRemote | null;
  source: string;
  locked: boolean;
  legacyKey?: string | null;
}

export function useUIDraft({
  namespace,
  target,
  remote,
  source,
  locked,
  legacyKey = null,
}: Args) {
  const stable = !!target && !!remote && !locked;
  const [generation, setGeneration] = useState(1);
  const [reloadFrom, setReloadFrom] = useState<number | null>(null);
  const [snapshot, setSnapshot] = useState<DraftSnapshot>(() => ({
    state:
      !stable && !locked && legacyKey ? readUIDraft(legacyKey, source) : {},
    revision: 0,
    writable: true,
    notice: "",
    key: 0,
  }));

  if (
    target &&
    remote &&
    !locked &&
    remote.settled &&
    (reloadFrom === null || remote.version > reloadFrom) &&
    snapshot.key !== generation
  )
    setSnapshot(
      restoreDraft({
        saved: remote.data,
        cached: readDraftCache(target.cacheKey),
        source,
        key: generation,
      }),
    );

  const ready = !stable || snapshot.key === generation;
  const superseded =
    stable && ready && snapshot.writable && remote?.data?.writable === false;
  const writable = snapshot.writable && !superseded;

  const writer = useDraftSave({
    target: stable ? target : null,
    source,
    epoch: snapshot.key,
    baseRevision: snapshot.revision,
    enabled: stable && ready && writable,
  });

  function update(state: Record<string, unknown>) {
    if (locked || !ready || !writable) return;
    if (stable) writer.save(persistedState(source, state));
    else if (legacyKey) saveUIDraft(legacyKey, source, state);
  }

  function reload() {
    if (!target || !remote || locked) return;
    writer.reset();
    removeDraftCache(target.cacheKey);
    setReloadFrom(remote.version);
    setGeneration((value) => value + 1);
    void remote.refetch();
  }

  function download() {
    downloadText(
      target?.downloadName ?? "view-draft.json",
      JSON.stringify(writer.latest.current ?? snapshot.state, null, 2),
      "application/json",
    );
  }

  return {
    namespace,
    initialState: snapshot.state,
    ready,
    writable,
    notice: superseded ? SUPERSEDED_NOTICE : snapshot.notice,
    revision: snapshot.key,
    onStateUpdate: update,
    syncError: writer.error,
    syncConflict: writer.conflict,
    saving: writer.saving,
    retrySync: writer.retry,
    flushSync: writer.flushNow,
    reloadDraft: reload,
    downloadDraft: download,
  };
}
