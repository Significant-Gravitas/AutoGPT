import React, { createContext, useContext } from "react";

interface ArrayItemContextValue {
  isArrayItem: boolean;
}

const ArrayItemContext = createContext<ArrayItemContextValue>({
  isArrayItem: false,
});

export const ArrayItemProvider: React.FC<{
  children: React.ReactNode;
}> = ({ children }) => {
  return (
    <ArrayItemContext.Provider value={{ isArrayItem: true }}>
      {children}
    </ArrayItemContext.Provider>
  );
};

export const useIsArrayItem = (): boolean => {
  const context = useContext(ArrayItemContext);
  return context.isArrayItem;
};
