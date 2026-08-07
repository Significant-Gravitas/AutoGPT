"use client";

import { createContext, ReactNode, useContext } from "react";
import { useTallyPopup } from "./useTallyPopup";

type TallyPopupState = ReturnType<typeof useTallyPopup>;

interface Props {
  children: ReactNode;
  enabled: boolean;
}

const TallyPopupContext = createContext<
  (TallyPopupState & { enabled: boolean }) | null
>(null);

export function TallyPopupProvider({ children, enabled }: Props) {
  const popup = useTallyPopup(enabled);
  return (
    <TallyPopupContext.Provider value={{ ...popup, enabled }}>
      {children}
    </TallyPopupContext.Provider>
  );
}

export function useTallyPopupContext() {
  return useContext(TallyPopupContext);
}
