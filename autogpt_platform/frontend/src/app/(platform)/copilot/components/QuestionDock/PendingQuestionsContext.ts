"use client";

import { createContext } from "react";
import type { PendingQuestions } from "./helpers";

/** Latest unanswered clarifying questions for the session, provided by
 *  ChatContainer so the tool chain can render the answer form inline on
 *  the matching ask_question rows. */
export const PendingQuestionsContext = createContext<PendingQuestions | null>(
  null,
);
