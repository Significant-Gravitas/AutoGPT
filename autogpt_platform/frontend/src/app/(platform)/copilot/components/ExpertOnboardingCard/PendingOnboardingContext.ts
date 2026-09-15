"use client";

import { createContext } from "react";

/** Tool-call id of the onboarding card the session is still waiting on,
 *  provided by ChatContainer. `null` — the default on surfaces that don't
 *  provide it, such as the share viewer — renders every card as history. */
export const PendingOnboardingContext = createContext<string | null>(null);
