"use client";

import { createContext } from "react";
import type { Decision } from "../../cardDecisions";

export interface AnsweredCardState {
  verdict: Decision;
  /** When the row holding the decision was saved; null until it is. */
  at: string | null;
  /** The row that holds the decision. */
  messageID: string;
}

/** Proposal cards the chat already answered, by confirmation id. A card
 *  reads its state from here, so it never asks again once the chat holds
 *  the answer, whichever tab or session gave it. */
export const AnsweredCardsContext = createContext<
  ReadonlyMap<string, AnsweredCardState>
>(new Map());
