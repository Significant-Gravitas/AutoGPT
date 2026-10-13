import { useEffect, useRef, useState } from "react";
import { usePutV2SaveUiDraft } from "@/app/api/__generated__/endpoints/chat/chat";
import type { UIDraftWrite } from "@/app/api/__generated__/models/uIDraftWrite";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { writeDraftCache } from "./draftCache";
import { useQueryClient } from "@tanstack/react-query";

interface Props {
  sessionId: string;
  viewId: string;
  toolCallId: string;
  source: string;
  cacheKey: string | null;
  ready: boolean;
  writable: boolean;
  revision: number;
}
export function useDraftSave(props: Props) {
  const queryClient = useQueryClient();
  const mutation = usePutV2SaveUiDraft();
  const revision = useRef(props.revision);
  const latest = useRef<Record<string, unknown> | null>(null);
  const lastSaved = useRef("");
  const busy = useRef(false);
  const mounted = useRef(true);
  const timer = useRef<ReturnType<typeof setTimeout>>();
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const [saving, setSaving] = useState(false);
  useEffect(() => {
    revision.current = props.revision;
    setConflict(false);
    setError("");
    lastSaved.current = "";
  }, [props.revision]);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      clearTimeout(timer.current);
    };
  }, []);
  useEffect(() => {
    if (!props.ready || !props.writable) clearTimeout(timer.current);
  }, [props.ready, props.writable]);

  async function flush() {
    if (
      !props.ready ||
      !props.writable ||
      busy.current ||
      conflict ||
      !latest.current
    )
      return;
    const state = latest.current;
    const serialized = JSON.stringify(state);
    if (serialized === lastSaved.current) return;
    if (new TextEncoder().encode(serialized).length > 240_000) {
      setError(
        "This draft is too large to sync. Download your data before leaving.",
      );
      return;
    }
    busy.current = true;
    setSaving(true);
    setError("");
    try {
      const result = await mutation.mutateAsync({
        sessionId: props.sessionId,
        viewId: props.viewId,
        data: {
          tool_call_id: props.toolCallId,
          expected_revision: revision.current,
          state: state as UIDraftWrite["state"],
        },
      });
      if (result.status !== 200) throw new Error("save");
      revision.current = result.data.revision;
      lastSaved.current = serialized;
      if (mounted.current)
        writeDraftCache(props.cacheKey, {
          source: props.source,
          state: latest.current ?? state,
          revision: revision.current,
          dirty: JSON.stringify(latest.current ?? state) !== serialized,
        });
      void queryClient.invalidateQueries({
        predicate: (query) =>
          query.queryKey[0] === "ui-draft" &&
          query.queryKey[2] === props.sessionId &&
          query.queryKey[3] === props.viewId,
      });
    } catch (cause) {
      if (mounted.current) {
        setConflict(cause instanceof ApiError && cause.status === 409);
        setError(
          cause instanceof ApiError && cause.status === 409
            ? "A newer draft exists. Your edits are still here; reload the saved draft before editing further."
            : "Draft sync failed. Your edits are still here. Retry before leaving this device.",
        );
      }
    } finally {
      busy.current = false;
      if (mounted.current) setSaving(false);
    }
    if (
      mounted.current &&
      lastSaved.current === serialized &&
      latest.current &&
      JSON.stringify(latest.current) !== serialized
    )
      timer.current = setTimeout(() => void flush(), 500);
  }
  function save(state: Record<string, unknown>) {
    latest.current = state;
    writeDraftCache(props.cacheKey, {
      source: props.source,
      state,
      revision: revision.current,
      dirty: JSON.stringify(state) !== lastSaved.current,
    });
    clearTimeout(timer.current);
    timer.current = setTimeout(() => void flush(), 800);
  }
  return { save, retry: flush, error, conflict, saving, latest };
}
