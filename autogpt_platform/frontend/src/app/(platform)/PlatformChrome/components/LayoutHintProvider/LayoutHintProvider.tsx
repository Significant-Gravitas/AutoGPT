"use client";

import { createContext, ReactNode, useContext } from "react";

import type { LayoutHint } from "../../helpers";

const LayoutHintContext = createContext<LayoutHint | undefined>(undefined);

interface Props {
  hint: LayoutHint | undefined;
  children: ReactNode;
}

// Carries the server-read layout cookie down to `usePlatformChrome`, so the
// server and the first client paint agree on which shell to render.
export function LayoutHintProvider({ hint, children }: Props) {
  return (
    <LayoutHintContext.Provider value={hint}>
      {children}
    </LayoutHintContext.Provider>
  );
}

export function useLayoutHint() {
  return useContext(LayoutHintContext);
}
