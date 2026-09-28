import { createContext } from "react";

/** The session the thread belongs to, for rows that need to look their own
 *  review up (a hand-off held for approval renders its card on the wire). */
export const ChatSessionContext = createContext<string | null>(null);
