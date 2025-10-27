import { create } from "zustand";

interface GraphStore {
  isGraphRunning: boolean;
  setIsGraphRunning: (isGraphRunning: boolean) => void;
}

export const useGraphStore = create<GraphStore>((set) => ({
  isGraphRunning: false,
  setIsGraphRunning: (isGraphRunning: boolean) => set({ isGraphRunning }),
}));
