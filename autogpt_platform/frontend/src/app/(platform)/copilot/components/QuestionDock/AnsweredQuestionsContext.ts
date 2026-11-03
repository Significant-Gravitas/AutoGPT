"use client";

import { createContext } from "react";
import type { AnsweredQuestions } from "./answeredQuestions";

/** What the user answered to the question cards they have moved past,
 *  provided by PendingAnswerContexts so each card can show its answers. */
export const AnsweredQuestionsContext = createContext<AnsweredQuestions>(
  new Map(),
);
