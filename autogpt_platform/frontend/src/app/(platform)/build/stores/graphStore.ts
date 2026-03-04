import { create } from "zustand";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { GraphMeta } from "@/app/api/__generated__/models/graphMeta";

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
    }),
}));
