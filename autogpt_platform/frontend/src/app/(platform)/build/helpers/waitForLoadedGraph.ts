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

/**
 * The saved graph the canvas shows, if it shows `graphID` (at `version`, when
 * given). Otherwise the canvas holds something else: the workflow as it was
 * before a save, or nothing yet while the page opens.
 */
export function getLoadedGraph(graphID: string, version: number | null) {
  const graph = useGraphStore.getState().loadedGraph;
  return isGraph(graph, graphID, version) ? graph : null;
}

export function waitForLoadedGraph(graphID: string, version: number | null) {
  return new Promise<GraphModel>((resolve, reject) => {
    const timer = setTimeout(() => {
      unsubscribe();
      reject(new GraphStillLoadingError());
    }, LOAD_TIMEOUT_MS);
    const unsubscribe = useGraphStore.subscribe(({ loadedGraph }) => {
      if (!isGraph(loadedGraph, graphID, version)) return;
      clearTimeout(timer);
      unsubscribe();
      resolve(loadedGraph);
    });
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
