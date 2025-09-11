import { create } from "zustand";

interface CustomNodeState {
  nodeAdvancedStates: Record<string, boolean>;
  toggleAdvanced: (nodeId: string) => void;
  setShowAdvanced: (nodeId: string, show: boolean) => void;
  getShowAdvanced: (nodeId: string) => boolean;
}

export const useCustomNodeStore = create<CustomNodeState>((set, get) => ({
  nodeAdvancedStates: {},
  toggleAdvanced: (nodeId: string) =>
    set((state) => ({
      nodeAdvancedStates: {
        ...state.nodeAdvancedStates,
        [nodeId]: !state.nodeAdvancedStates[nodeId],
      },
    })),
  setShowAdvanced: (nodeId: string, show: boolean) =>
    set((state) => ({
      nodeAdvancedStates: {
        ...state.nodeAdvancedStates,
        [nodeId]: show,
      },
    })),
  getShowAdvanced: (nodeId: string) =>
    get().nodeAdvancedStates[nodeId] || false,
}));
