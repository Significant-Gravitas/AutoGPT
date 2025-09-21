import { createContext } from "react";

export const ArrayEditorContext = createContext<{
  isArrayItem: boolean;
  fieldKey: string;
  isConnected: boolean;
}>({
  isArrayItem: false,
  fieldKey: "",
  isConnected: false,
});
