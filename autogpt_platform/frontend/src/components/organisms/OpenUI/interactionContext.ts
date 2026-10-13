import { createContext, useContext } from "react";

export const OpenUIInteractionContext = createContext(false);
export const OpenUIInstanceContext = createContext("");

export function useOpenUIDisabled() {
  return useContext(OpenUIInteractionContext);
}
