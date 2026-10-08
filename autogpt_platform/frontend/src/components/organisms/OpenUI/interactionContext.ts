import { createContext, useContext } from "react";

export const OpenUIInteractionContext = createContext(false);

export function useOpenUIDisabled() {
  return useContext(OpenUIInteractionContext);
}
