import { create } from "zustand";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";

// The workflow and version the builder is loading. `forLatest` when the URL
// asked for no version, and `latestVersion` once that request has resolved:
// the builder then writes that version into the URL, which keeps the request.
type LoadRequest = {
  graphID: string | null;
  version: number | null;
  forLatest: boolean;
  latestVersion: number | null;
};

const NO_LOAD_REQUEST: LoadRequest = {
  graphID: null,
  version: null,
  forLatest: true,
  latestVersion: null,
};

let loadController = new AbortController();

function renewLoadSignal() {
  loadController.abort();
  loadController = new AbortController();
  return loadController.signal;
}

interface GraphStore {
  graphExecutionStatus: AgentExecutionStatus | undefined;
  isGraphRunning: boolean;
  setGraphExecutionStatus: (status: AgentExecutionStatus | undefined) => void;
  setIsGraphRunning: (isRunning: boolean) => void;

  inputSchema: Record<string, any> | null;
  credentialsInputSchema: Record<string, any> | null;
  outputSchema: Record<string, any> | null;

  setGraphSchemas: (
    inputSchema: Record<string, any> | null,
    credentialsInputSchema: Record<string, any> | null,
    outputSchema: Record<string, any> | null,
  ) => void;

  // Available graphs; used for sub-graph updated version detection
  availableSubGraphs: GraphMeta[];
  setAvailableSubGraphs: (graphs: GraphMeta[]) => void;

  // Only useFlow writes the load fields below.
  // The saved graph the canvas was last loaded from, which a save of a saved
  // workflow becomes a new version of, and the version the URL asked for then
  // (null for the latest).
  loadedGraph: GraphModel | null;
  loadedForVersion: number | null;
  setLoadedGraph: (graph: GraphModel | null) => void;
  loadRequest: LoadRequest;
  requestLoad: (graphID: string | null, version: number | null) => void;
  resolveLatestVersion: (graphID: string, version: number) => void;
  // Aborted when the builder moves to another workflow or version, or closes,
  // so a save waiting for the canvas to load gives up.
  loadSignal: AbortSignal;
  // The canvas can't load the workflow or its blocks.
  loadFailed: boolean;
  setLoadFailed: (failed: boolean) => void;

  hasInputs: () => boolean;
  hasCredentials: () => boolean;
  hasOutputs: () => boolean;
  // Also aborts loadSignal, which cancels any save waiting for the canvas.
  reset: () => void;
}

export const useGraphStore = create<GraphStore>((set, get) => ({
  graphExecutionStatus: undefined,
  isGraphRunning: false,
  inputSchema: null,
  credentialsInputSchema: null,
  outputSchema: null,
  availableSubGraphs: [],
  loadedGraph: null,
  loadedForVersion: null,
  loadRequest: NO_LOAD_REQUEST,
  loadSignal: loadController.signal,
  loadFailed: false,

  setGraphExecutionStatus: (status: AgentExecutionStatus | undefined) => {
    set({
      graphExecutionStatus: status,
      isGraphRunning:
        status === AgentExecutionStatus.RUNNING ||
        status === AgentExecutionStatus.QUEUED,
    });
  },

  setIsGraphRunning: (isRunning: boolean) => {
    set({ isGraphRunning: isRunning });
  },

  setGraphSchemas: (inputSchema, credentialsInputSchema, outputSchema) =>
    set({ inputSchema, credentialsInputSchema, outputSchema }),

  setAvailableSubGraphs: (graphs) => set({ availableSubGraphs: graphs }),

  setLoadedGraph: (graph) =>
    set(({ loadRequest }) => ({
      loadedGraph: graph,
      loadedForVersion: loadRequest.forLatest ? null : loadRequest.version,
    })),

  requestLoad: (graphID, version) => {
    const request = get().loadRequest;
    if (request.graphID === graphID && request.version === version) return;
    const getsItsLatestVersion =
      request.graphID === graphID &&
      request.version === null &&
      version !== null &&
      version === request.latestVersion;
    if (getsItsLatestVersion) {
      set({ loadRequest: { ...request, version } });
      return;
    }
    set({
      loadRequest: {
        graphID,
        version,
        forLatest: version === null,
        latestVersion: null,
      },
      loadSignal: renewLoadSignal(),
    });
  },

  resolveLatestVersion: (graphID, version) => {
    const request = get().loadRequest;
    if (request.graphID !== graphID || request.version !== null) return;
    set({ loadRequest: { ...request, latestVersion: version } });
  },

  setLoadFailed: (failed) => set({ loadFailed: failed }),

  hasOutputs: () => {
    const { outputSchema } = get();
    return Object.keys(outputSchema?.properties ?? {}).length > 0;
  },

  hasInputs: () => {
    const { inputSchema } = get();
    return Object.keys(inputSchema?.properties ?? {}).length > 0;
  },

  hasCredentials: () => {
    const { credentialsInputSchema } = get();
    return Object.keys(credentialsInputSchema?.properties ?? {}).length > 0;
  },

  reset: () =>
    set({
      graphExecutionStatus: undefined,
      isGraphRunning: false,
      inputSchema: null,
      credentialsInputSchema: null,
      loadedGraph: null,
      loadedForVersion: null,
      loadRequest: NO_LOAD_REQUEST,
      loadSignal: renewLoadSignal(),
      loadFailed: false,
    }),
}));
