import { createContext } from "react";

export const ArrayEditorContext = createContext<{
  isArrayItem: boolean;
  arrayFieldHandleId: string;
  isConnected: boolean;
}>({
  isArrayItem: false,
  arrayFieldHandleId: "",
  isConnected: false,
});
