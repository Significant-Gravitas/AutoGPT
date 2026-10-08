import { useEffect, useRef, useState } from "react";
import { parseWorkspace } from "@/lib/openui/parse";
import { streamSample } from "./helpers";

export function useWorkspaceGeneration(initialSource: string) {
  const [source, setSource] = useState(initialSource);
  const [isStreaming, setIsStreaming] = useState(false);
  const [revision, setRevision] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const active = useRef<AbortController | null>(null);
  const committed = useRef(initialSource);
  useEffect(() => () => active.current?.abort(), []);

  function stop() {
    active.current?.abort();
    active.current = null;
    setSource(committed.current);
    setIsStreaming(false);
  }

  function reset(nextSource: string) {
    stop();
    committed.current = nextSource;
    setSource(nextSource);
    setError(null);
    setRevision((value) => value + 1);
  }

  async function generate(sample: string) {
    active.current?.abort();
    const controller = new AbortController();
    active.current = controller;
    setError(null);
    setIsStreaming(true);
    setRevision((value) => value + 1);
    setSource("");
    function update(value: string) {
      if (active.current === controller) setSource(value);
    }
    try {
      const next = await streamSample(sample, controller.signal, update);
      parseWorkspace(next);
      if (active.current !== controller) return false;
      committed.current = next;
      setSource(next);
      return true;
    } catch (reason) {
      if (active.current === controller) {
        setSource(committed.current);
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
    isStreaming,
    revision,
    error,
    setError,
    generate,
    reset,
    stop,
  };
}
