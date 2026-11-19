import { GraphModel } from "@/app/api/__generated__/models/graphModel";
import { useGraphStore } from "../stores/graphStore";

// Loading takes the graph's request and then its blocks' request, each of
// which retries up to three times.
const LOAD_TIMEOUT_MS = 30_000;

class GraphStillLoadingError extends Error {
  constructor() {
    super("Your workflow is still loading. Try again in a moment.");
    this.name = "GraphStillLoadingError";
  }
}

class GraphLoadFailedError extends Error {
  constructor() {
    super("Your workflow couldn't be loaded. Reload the page and try again.");
    this.name = "GraphLoadFailedError";
  }
}

// The builder moved to another workflow or closed before this one loaded.
class SaveCancelledError extends Error {
  constructor() {
    super("The save was cancelled.");
    this.name = "SaveCancelledError";
  }
}

export function isSaveCancelled(error: unknown) {
  return error instanceof SaveCancelledError;
}

/**
 * The saved graph the canvas shows, if it shows `graphID` (at `version`, when
 * given). Otherwise the canvas holds something else: the workflow as it was
 * before a save, or nothing yet while the page opens.
 */
export function getLoadedGraph(graphID: string, version: number | null) {
  const graph = useGraphStore.getState().loadedGraph;
  return isGraph(graph, graphID, version) ? graph : null;
}

/** Whether the canvas shows a saved graph other than this version of `graph`. */
export function showsAnotherSavedGraph(graph: GraphModel) {
  const loaded = useGraphStore.getState().loadedGraph;
  return (
    loaded !== null && !isGraph(loaded, graph.id ?? "", graph.version ?? null)
  );
}

/**
 * Resolves once the canvas shows `graphID` (at `version`, when given). Fails at
 * once if its blocks can't load, and is cancelled (see `isSaveCancelled`) if
 * the builder moves to another workflow or closes first.
 */
export function waitForLoadedGraph(graphID: string, version: number | null) {
  return new Promise<GraphModel>((resolve, reject) => {
    const { loadSignal } = useGraphStore.getState();
    const timer = setTimeout(
      () => settle(new GraphStillLoadingError()),
      LOAD_TIMEOUT_MS,
    );
    const onAbort = () => settle(new SaveCancelledError());
    const unsubscribe = useGraphStore.subscribe(check);
    loadSignal.addEventListener("abort", onAbort);
    if (loadSignal.aborted) onAbort();
    else check(useGraphStore.getState());

    function check(state: ReturnType<typeof useGraphStore.getState>) {
      if (isGraph(state.loadedGraph, graphID, version)) {
        settle(null, state.loadedGraph);
      } else if (state.loadFailed) {
        settle(new GraphLoadFailedError());
      }
    }

    function settle(error: Error | null, graph?: GraphModel) {
      clearTimeout(timer);
      unsubscribe();
      loadSignal.removeEventListener("abort", onAbort);
      if (graph) resolve(graph);
      else reject(error);
    }
  });
}

function isGraph(
  graph: GraphModel | null,
  graphID: string,
  version: number | null,
): graph is GraphModel {
  return (
    graph?.id === graphID && (version === null || graph.version === version)
  );
}
