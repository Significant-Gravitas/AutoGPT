import { create } from "zustand";

interface GraphStore {
  isGraphRunning: boolean;
  setIsGraphRunning: (isGraphRunning: boolean) => void;

  inputSchema: Record<string, any> | null;
  credentialsInputSchema: Record<string, any> | null;
  setGraphSchemas: (
    inputSchema: Record<string, any> | null,
    credentialsInputSchema: Record<string, any> | null,
  ) => void;

  hasInputs: () => boolean;
  hasCredentials: () => boolean;
  reset: () => void;
}

export const useGraphStore = create<GraphStore>((set, get) => ({
  isGraphRunning: false,
  inputSchema: null,
  credentialsInputSchema: null,

  setIsGraphRunning: (isGraphRunning: boolean) => set({ isGraphRunning }),

  setGraphSchemas: (inputSchema, credentialsInputSchema) =>
    set({ inputSchema, credentialsInputSchema }),

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
      isGraphRunning: false,
      inputSchema: null,
      credentialsInputSchema: null,
    }),
}));
