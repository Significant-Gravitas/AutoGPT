import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { AppEnv, environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";
import type { UIMessage } from "ai";
import { toast as sonnerToast } from "sonner";
import { isKickoffMessage } from "../expertKickoff";

const GITHUB_REPO_URL = "https://github.com/Significant-Gravitas/AutoGPT";
const GITHUB_STAR_TOAST_KEY = "github-star";
const GITHUB_STAR_TOAST_DURATION_MS = 15_000;
const ANSWERS_BEFORE_STAR_TOAST = 5;

interface FinishedTurn {
  messages: UIMessage[];
  isAbort?: boolean;
  isDisconnect?: boolean;
  isError?: boolean;
}

// A continuation turn the backend starts after an answer finishes still ends
// on the prompt that answer counted, so each prompt counts once.
const countedPrompts = new Set<string>();

/** Self-hosted installs (app env `local`, never dev or prod) ask for a GitHub
 *  star once Otto has answered five of the user's prompts, in any chat. A turn
 *  that failed, was stopped or dropped counts for nothing, so the ask never
 *  lands beside an error; nor does an expert's kickoff, which the app sends
 *  itself. The count stops at five, so the toast never comes back. */
export function countAnswerTowardStarToast(turn: FinishedTurn) {
  if (environment.getAppEnv() !== AppEnv.LOCAL) return;
  if (turn.isAbort || turn.isDisconnect || turn.isError) return;
  const prompt = turn.messages.findLast((message) => message.role === "user");
  if (!prompt || isKickoffMessage(prompt) || countedPrompts.has(prompt.id))
    return;
  const answered = (Number(storage.get(Key.GITHUB_STAR_PROMPT_COUNT)) || 0) + 1;
  if (answered > ANSWERS_BEFORE_STAR_TOAST) return;
  countedPrompts.add(prompt.id);
  storage.set(Key.GITHUB_STAR_PROMPT_COUNT, String(answered));
  if (answered < ANSWERS_BEFORE_STAR_TOAST) return;
  toast({
    title: "Enjoying AutoGPT?",
    description: "A star on GitHub helps more people find the project.",
    duration: GITHUB_STAR_TOAST_DURATION_MS,
    replaceKey: GITHUB_STAR_TOAST_KEY,
    action: <StarOnGitHubButton />,
  });
}

/** Forgets the prompts counted so far, for tests that reuse message ids. */
export function resetStarToastCounts() {
  countedPrompts.clear();
}

function StarOnGitHubButton() {
  return (
    <Button
      as="NextLink"
      href={GITHUB_REPO_URL}
      target="_blank"
      rel="noopener noreferrer"
      size="small"
      variant="secondary"
      onClick={() => sonnerToast.dismiss(GITHUB_STAR_TOAST_KEY)}
    >
      Star on GitHub
      <span className="sr-only"> (opens in a new tab)</span>
    </Button>
  );
}
