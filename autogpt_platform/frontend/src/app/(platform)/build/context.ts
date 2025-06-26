import { createContext } from "react";

export type FlowContextType = {
  visualizeBeads: "no" | "static" | "animate";
  setIsAnyModalOpen: (isOpen: boolean) => void;
  getNextNodeId: () => string;
};

export const FlowContext = createContext<FlowContextType | null>(null);
