"use client";

import { createContext } from "react";

/** The `data` of the pause the chat is sitting on, or `null` when it sits on
 *  none, provided by PendingAnswerContexts. `undefined` — the default on
 *  surfaces that don't track it, such as the share viewer — says nothing
 *  either way, so every card there shows just its title. */
export const CurrentPauseContext = createContext<unknown>(undefined);
