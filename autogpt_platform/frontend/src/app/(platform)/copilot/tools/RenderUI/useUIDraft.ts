import { useEffect, useRef, useState } from "react";
import { useGetV2GetUiDraft } from "@/app/api/__generated__/endpoints/chat/chat";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { parseWorkspace } from "@/lib/openui/parse";
import { mergeDraft } from "@/lib/openui/draft-merge";
import { persistedState } from "@/lib/openui/draft-fields";
import { downloadText } from "@/lib/openui/download";
import { readUIDraft, saveUIDraft } from "./helpers";
import { readDraftCache } from "./draftCache";
import { useDraftSave } from "./useDraftSave";

export function useUIDraft(
  draftID: string | null,
  source: string,
  locked: boolean,
) {
  const ownerId = useAuthStore((state) => state.user?.id);
  const [sessionId = "", toolCallId = ""] = draftID?.split(":") ?? [];
  let viewId = "";
  try {
    viewId = String(parseWorkspace(source).props.viewId ?? "");
  } catch {}
  const stable = !!(viewId && draftID && !locked);
  const cacheKey =
    stable && ownerId
      ? `copilot-ui:v2:${ownerId}:${sessionId}:${viewId}`
      : null;
  const legacyKey = draftID ? `copilot-ui:${draftID}` : null;
  const query = useGetV2GetUiDraft(
    sessionId,
    viewId,
    { tool_call_id: toolCallId },
    {
      query: {
        enabled: stable,
        retry: false,
        refetchOnWindowFocus: false,
        queryKey: ["ui-draft", ownerId, sessionId, viewId, toolCallId],
      },
    },
  );
  const applied = useRef(false);
  const [snapshot, setSnapshot] = useState(() => ({
    state:
      !stable && !locked && legacyKey ? readUIDraft(legacyKey, source) : {},
    revision: 0,
    writable: true,
    notice: "",
    ready: !stable,
    key: 0,
  }));
  const writer = useDraftSave({
    sessionId,
    viewId,
    toolCallId,
    source,
    cacheKey,
    ready: snapshot.ready && stable,
    writable: snapshot.writable && !locked,
    revision: snapshot.revision,
  });

  useEffect(() => {
    if (
      query.data?.status === 200 &&
      !query.data.data.writable &&
      applied.current
    )
      setSnapshot((value) =>
        value.writable
          ? {
              ...value,
              writable: false,
              notice:
                "A newer version of this view is available below. This older version is read-only.",
            }
          : value,
      );
  }, [query.data]);

  useEffect(() => {
    if (!stable || applied.current || query.isPending || query.isFetching)
      return;
    applied.current = true;
    const saved = query.data?.status === 200 ? query.data.data : null;
    const cached = readDraftCache(cacheKey);
    const useCache =
      cached &&
      (!saved ||
        (cached.source === source &&
          saved.writable &&
          (cached.dirty !== false || cached.revision >= saved.revision)));
    const base = useCache ? cached : saved;
    try {
      const merged = base
        ? mergeDraft(base.source, source, base.state)
        : { state: {}, revised: 0 };
      setSnapshot({
        state: saved?.writable === false ? {} : merged.state,
        revision: useCache ? cached.revision : (saved?.revision ?? 0),
        writable: saved?.writable !== false,
        ready: true,
        key: 1,
        notice:
          saved?.writable === false
            ? "A newer version of this view is available below. This older version is read-only."
            : !saved
              ? "Saved draft could not be loaded. Your changes will stay here until sync is available."
              : merged.revised
                ? `Updated ${merged.revised} field${merged.revised === 1 ? "" : "s"} from the revised plan; unchanged edits were kept.`
                : "",
      });
    } catch {
      setSnapshot({
        state: {},
        revision: saved?.revision ?? 0,
        writable: saved?.writable !== false,
        ready: true,
        key: 1,
        notice:
          "The previous draft could not be restored. The current view is shown.",
      });
    }
  }, [cacheKey, query.data, query.isPending, query.isFetching, source, stable]);

  function update(state: Record<string, unknown>) {
    if (locked || !snapshot.ready || !snapshot.writable) return;
    if (stable) writer.save(persistedState(source, state));
    else if (legacyKey) saveUIDraft(legacyKey, source, state);
  }
  async function reload() {
    applied.current = false;
    setSnapshot((value) => ({ ...value, ready: false }));
    if (cacheKey) {
      try {
        localStorage.removeItem(cacheKey);
      } catch {}
    }
    await query.refetch();
  }
  function download() {
    downloadText(
      `${viewId || "view"}-draft.json`,
      JSON.stringify(writer.latest.current ?? snapshot.state, null, 2),
      "application/json",
    );
  }
  return {
    namespace: `${sessionId}:${viewId}`,
    initialState: snapshot.state,
    ready: snapshot.ready,
    writable: snapshot.writable,
    notice: snapshot.notice,
    revision: snapshot.key,
    onStateUpdate: update,
    syncError: writer.error,
    syncConflict: writer.conflict,
    saving: writer.saving,
    retrySync: writer.retry,
    reloadDraft: reload,
    downloadDraft: download,
  };
}
