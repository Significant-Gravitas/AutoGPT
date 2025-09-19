import { createContext } from "react";

export const HandleContext = createContext<{
  isArrayItem: boolean;
  fieldKey: string;
  isConnected: boolean;
}>({
  isArrayItem: false,
  fieldKey: "",
  isConnected: false,
});
