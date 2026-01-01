import React, { createContext, useContext } from "react";

interface ArrayItemContextValue {
  isArrayItem: boolean;
  arrayItemHandleId: string;
}

const ArrayItemContext = createContext<ArrayItemContextValue>({
  isArrayItem: false,
  arrayItemHandleId: "",
});

export const ArrayItemProvider: React.FC<{
  children: React.ReactNode;
  arrayItemHandleId: string;
}> = ({ children, arrayItemHandleId }) => {
  return (
    <ArrayItemContext.Provider value={{ isArrayItem: true, arrayItemHandleId }}>
      {children}
    </ArrayItemContext.Provider>
  );
};

export const useIsArrayItem = (): boolean => {
  // here this will be true if field is inside an array
  const context = useContext(ArrayItemContext);
  return context.isArrayItem;
};

export const useArrayItemHandleId = (): string => {
  const context = useContext(ArrayItemContext);
  return context.arrayItemHandleId;
};
