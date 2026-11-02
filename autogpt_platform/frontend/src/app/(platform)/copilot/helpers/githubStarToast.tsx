import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { AppEnv, environment } from "@/services/environment";
import { Key, storage } from "@/services/storage/local-storage";
import { toast as sonnerToast } from "sonner";

const GITHUB_REPO_URL = "https://github.com/Significant-Gravitas/AutoGPT";
const GITHUB_STAR_TOAST_KEY = "github-star";
const GITHUB_STAR_TOAST_DURATION_MS = 15_000;
const PROMPTS_BEFORE_STAR_TOAST = 5;

/** Self-hosted installs (app env `local`, never dev or prod) ask for a GitHub
 *  star once, when the fifth prompt typed into any chat has gone out. The
 *  count stops at five, so the toast never comes back. */
export function countPromptTowardStarToast() {
  if (environment.getAppEnv() !== AppEnv.LOCAL) return;
  const sent = (Number(storage.get(Key.GITHUB_STAR_PROMPT_COUNT)) || 0) + 1;
  if (sent > PROMPTS_BEFORE_STAR_TOAST) return;
  storage.set(Key.GITHUB_STAR_PROMPT_COUNT, String(sent));
  if (sent < PROMPTS_BEFORE_STAR_TOAST) return;
  toast({
    title: "Enjoying AutoGPT?",
    description: "A star on GitHub helps more people find the project.",
    duration: GITHUB_STAR_TOAST_DURATION_MS,
    replaceKey: GITHUB_STAR_TOAST_KEY,
    action: <StarOnGitHubButton />,
  });
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
    </Button>
  );
}
