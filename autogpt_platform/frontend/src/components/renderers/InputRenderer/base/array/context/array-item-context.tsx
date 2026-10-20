import { createContext, useContext } from "react";

interface ArrayItemContextValue {
  isArrayItem: boolean;
  arrayItemHandleId: string;
}

const ArrayItemContext = createContext<ArrayItemContextValue>({
  isArrayItem: false,
  arrayItemHandleId: "",
});

export const useIsArrayItem = (): boolean => {
  // here this will be true if field is inside an array
  const context = useContext(ArrayItemContext);
  return context.isArrayItem;
};
