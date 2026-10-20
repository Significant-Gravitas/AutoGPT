"use client";

import { createContext } from "react";
import type { HeldOutcome } from "./heldCallRows";

/** Answered held calls by their original tool call id, so the chain row and
 *  the approval queue read the answer from the persisted late result. */
export const HeldOutcomesContext = createContext<
  ReadonlyMap<string, HeldOutcome>
>(new Map());
