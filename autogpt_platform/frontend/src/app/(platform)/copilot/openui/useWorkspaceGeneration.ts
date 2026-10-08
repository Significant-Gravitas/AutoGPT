import { useEffect, useRef, useState } from "react";
import { parseWorkspace } from "@/lib/openui/parse";
import { streamLiveWorkspace, streamSample } from "./helpers";

export function useWorkspaceGeneration(initialSource: string) {
  const [source, setSource] = useState(initialSource);
  const [sourceMode, setSourceMode] = useState<"sample" | "live">("sample");
  const [isStreaming, setIsStreaming] = useState(false);
  const [revision, setRevision] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const active = useRef<AbortController | null>(null);
  const committed = useRef(initialSource);
  const committedMode = useRef<"sample" | "live">("sample");
  useEffect(() => () => active.current?.abort(), []);

  function stop() {
    active.current?.abort();
    active.current = null;
    setSource(committed.current);
    setSourceMode(committedMode.current);
    setIsStreaming(false);
  }

  function reset(nextSource: string) {
    stop();
    committed.current = nextSource;
    committedMode.current = "sample";
    setSourceMode("sample");
    setSource(nextSource);
    setError(null);
    setRevision((value) => value + 1);
  }

  async function generate(
    prompt: string,
    sample: string | null,
    fields: Record<string, unknown> = {},
  ) {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setError(null);
    setIsStreaming(true);
    setRevision((value) => value + 1);
    setSource("");
    setSourceMode(sample === null ? "live" : "sample");
    function update(value: string) {
      if (active.current === controller) setSource(value);
    }
    try {
      const next =
        sample === null
          ? await streamLiveWorkspace(
              { prompt, source: committed.current, fields },
              controller.signal,
              update,
            )
          : await streamSample(sample, controller.signal, update);
      parseWorkspace(next);
      if (active.current !== controller) return false;
      committed.current = next;
      committedMode.current = sample === null ? "live" : "sample";
      setSource(next);
      return true;
    } catch (reason) {
      if (active.current === controller) {
        setSource(committed.current);
        setSourceMode(committedMode.current);
        if (!controller.signal.aborted)
          setError(
            reason instanceof Error
              ? reason.message
              : "Something went wrong. Please try again.",
          );
      }
      return false;
    } finally {
      if (active.current === controller) {
        active.current = null;
        setIsStreaming(false);
      }
      controller.abort();
    }
  }

  return {
    source,
    sourceMode,
    isStreaming,
    revision,
    error,
    setError,
    generate,
    reset,
    stop,
  };
}
