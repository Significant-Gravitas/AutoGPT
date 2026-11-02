import { useMountEffect } from "./useMountEffect";

const PROMPT_KEY = "importWorkflowPrompt";
const AUTOSUBMIT_KEY = "importWorkflowAutosubmit";
const PARAM = "triggerprompt";

/**
 * Capture a `?triggerprompt=` value handed off from the marketing site and
 * stash it into sessionStorage so it survives the signup → onboarding →
 * /copilot redirect chain (which would otherwise strip the URL). The /copilot
 * page picks it up via `useWorkflowImportAutoSubmit` and auto-submits.
 */
export function useCaptureMarketingPrompt() {
  useMountEffect(() => {
    if (typeof window === "undefined") return;

    const url = new URL(window.location.href);
    const prompt = url.searchParams.get(PARAM);
    if (!prompt || !prompt.trim()) return;

    sessionStorage.setItem(PROMPT_KEY, prompt.trim());
    sessionStorage.setItem(AUTOSUBMIT_KEY, "true");

    url.searchParams.delete(PARAM);
    window.history.replaceState(
      null,
      "",
      `${url.pathname}${url.search}${url.hash}`,
    );
  });
}
