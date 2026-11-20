import { stashCopilotPrompt } from "@/services/copilot/prompt-handoff";
import { useMountEffect } from "./useMountEffect";

const PARAM = "triggerprompt";

/**
 * Capture a `?triggerprompt=` value handed off from the marketing site and
 * stash it into sessionStorage so it survives the signup → onboarding →
 * /copilot redirect chain (which would otherwise strip the URL). The /copilot
 * page prefills it via `useWorkflowImportAutoSubmit`: the value comes from
 * another origin, so the user sends it.
 */
export function useCaptureMarketingPrompt() {
  useMountEffect(() => {
    if (typeof window === "undefined") return;

    const url = new URL(window.location.href);
    const prompt = url.searchParams.get(PARAM);
    if (!prompt || !prompt.trim()) return;

    stashCopilotPrompt({ prompt: prompt.trim(), autosubmit: false });

    url.searchParams.delete(PARAM);
    window.history.replaceState(
      null,
      "",
      `${url.pathname}${url.search}${url.hash}`,
    );
  });
}
