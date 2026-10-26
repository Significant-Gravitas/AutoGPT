"use client";

import { createContext } from "react";

/** Hand-offs a later one in this chat superseded, by tool call id. */
export const SupersededDelegationsContext = createContext<ReadonlySet<string>>(
  new Set(),
);
