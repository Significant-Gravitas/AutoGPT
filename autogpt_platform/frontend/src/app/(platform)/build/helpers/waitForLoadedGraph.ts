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

type GraphStoreState = ReturnType<typeof useGraphStore.getState>;

/**
 * The saved graph the canvas shows, if it shows `graphID` as the URL asks for
 * it: at `version`, or, with no version, as loaded for a URL without one.
 * Otherwise the canvas holds something else: the workflow as it was before a
 * save, an earlier version, or nothing yet while the page opens.
 */
export function getLoadedGraph(graphID: string, version: number | null) {
  return shownGraph(useGraphStore.getState(), graphID, version);
}

/** Whether the canvas was last loaded from this version of `graph`. */
export function showsSavedGraph(graph: GraphModel) {
  const loaded = useGraphStore.getState().loadedGraph;
  return (
    loaded !== null &&
    loaded.id === graph.id &&
    loaded.version === graph.version
  );
}

/** Whether the canvas shows a saved graph other than this version of `graph`. */
export function showsAnotherSavedGraph(graph: GraphModel) {
  return (
    useGraphStore.getState().loadedGraph !== null && !showsSavedGraph(graph)
  );
}

/**
 * Resolves once the canvas shows `graphID` as the URL asks for it (see
 * `getLoadedGraph`). Fails at once if the workflow or its blocks can't load,
 * and is cancelled (see `isSaveCancelled`) if the builder moves to another
 * workflow or closes first.
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

    function check(state: GraphStoreState) {
      const graph = shownGraph(state, graphID, version);
      if (graph) {
        settle(null, graph);
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

// A URL without a version asks for the latest one, which only a load made for
// such a URL is sure to show.
function shownGraph(
  { loadedGraph, loadedForVersion }: GraphStoreState,
  graphID: string,
  version: number | null,
) {
  if (loadedGraph?.id !== graphID) return null;
  const shows =
    version === null
      ? loadedForVersion === null
      : loadedGraph.version === version;
  return shows ? loadedGraph : null;
}
