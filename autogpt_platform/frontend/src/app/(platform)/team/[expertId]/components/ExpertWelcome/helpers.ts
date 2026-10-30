import type { RaiseIntroQuestion } from "@/app/api/__generated__/models/raiseIntroQuestion";

const DONE_KEY_PREFIX = "expert-welcome-done:";

export type WelcomeAnswers = Record<number, string>;

/** Whether this expert's opening questions were already answered or skipped
 *  in this browser, so a reload of the welcome link does not ask again. */
export function isWelcomeDone(expertId: string): boolean {
  try {
    return window.localStorage.getItem(DONE_KEY_PREFIX + expertId) !== null;
  } catch {
    return false;
  }
}

export function markWelcomeDone(expertId: string) {
  try {
    window.localStorage.setItem(DONE_KEY_PREFIX + expertId, "1");
  } catch {
    // Best-effort: without storage the questions can simply be asked again.
  }
}

export function isEveryQuestionAnswered(
  questions: RaiseIntroQuestion[],
  answers: WelcomeAnswers,
): boolean {
  return (
    questions.length > 0 &&
    questions.every((_, index) => Boolean(answers[index]?.trim()))
  );
}

export function toSoulAnswers(
  questions: RaiseIntroQuestion[],
  answers: WelcomeAnswers,
) {
  return questions.flatMap((question, index) => {
    const answer = answers[index]?.trim();
    return answer ? [{ question: question.question, answer }] : [];
  });
}
