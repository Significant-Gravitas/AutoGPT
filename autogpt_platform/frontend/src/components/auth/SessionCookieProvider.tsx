"use client";

import { createContext, useContext, type ReactNode } from "react";

// Whether the request had a session cookie, read on the server by the layouts
// of the sign-in pages. Outside them it isn't known, so a session may exist.
const SessionCookieContext = createContext(true);

interface Props {
  hasSessionCookie: boolean;
  children: ReactNode;
}

export function SessionCookieProvider({ hasSessionCookie, children }: Props) {
  return (
    <SessionCookieContext.Provider value={hasSessionCookie}>
      {children}
    </SessionCookieContext.Provider>
  );
}

export function useHasSessionCookie() {
  return useContext(SessionCookieContext);
}
