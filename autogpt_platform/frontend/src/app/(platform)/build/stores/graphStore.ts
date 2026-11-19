import { create } from "zustand";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";
import { GraphModel } from "@/app/api/__generated__/models/graphModel";

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

  // The saved graph the canvas was last loaded from, which a save of a saved
  // workflow becomes a new version of, and the version the URL asked for then
  // (null for the latest).
  loadedGraph: GraphModel | null;
  loadedForVersion: number | null;
  setLoadedGraph: (graph: GraphModel | null, forVersion: number | null) => void;
  // Aborted when the builder moves to another workflow or closes, so a save
  // waiting for the canvas to load gives up.
  loadSignal: AbortSignal;
  setLoadSignal: (signal: AbortSignal) => void;
  // The canvas can't load the workflow's blocks.
  loadFailed: boolean;
  setLoadFailed: (failed: boolean) => void;

  hasInputs: () => boolean;
  hasCredentials: () => boolean;
  hasOutputs: () => boolean;
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
  loadSignal: new AbortController().signal,
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

  setLoadedGraph: (graph, forVersion) =>
    set({ loadedGraph: graph, loadedForVersion: forVersion }),

  setLoadSignal: (signal) => set({ loadSignal: signal }),

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
      loadSignal: new AbortController().signal,
      loadFailed: false,
    }),
}));
