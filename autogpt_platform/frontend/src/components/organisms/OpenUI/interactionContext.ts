import { createContext, useContext } from "react";

export const OpenUIInteractionContext = createContext(false);
export const OpenUIActionsContext = createContext(false);
export const OpenUIInstanceContext = createContext("");

export function useOpenUIDisabled() {
  return useContext(OpenUIInteractionContext);
}

export function useOpenUIActionsDisabled() {
  const disabled = useContext(OpenUIInteractionContext);
  const actionsDisabled = useContext(OpenUIActionsContext);
  return disabled || actionsDisabled;
}
