import { COPILOT_PROMPT_KEYS } from "@/services/copilot/prompt-handoff";
import type { FileUIPart } from "ai";
import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "./store";

/**
 * Take the prompt the app stashed for this page, or the one in the URL hash
 * (/copilot#prompt=URL-encoded-text). Only a stashed prompt can ask to be
 * sent, through `stashCopilotPrompt`; a prompt from the URL is prefilled, so
 * sending it stays the user's own action. Returns null if there is none.
 */
function extractPromptFromUrl(): {
  prompt: string;
  autosubmit: boolean;
  filePart?: FileUIPart;
} | null {
  if (typeof window === "undefined") return null;

  // Check sessionStorage first (used by workflow import for large prompts
  // and by the marketing-site CTA handoff via useCaptureMarketingPrompt).
  const storedPrompt = sessionStorage.getItem(COPILOT_PROMPT_KEYS.prompt);
  if (storedPrompt) {
    sessionStorage.removeItem(COPILOT_PROMPT_KEYS.prompt);

    const autosubmit =
      sessionStorage.getItem(COPILOT_PROMPT_KEYS.autosubmit) === "true";
    sessionStorage.removeItem(COPILOT_PROMPT_KEYS.autosubmit);

    // Check for a pre-uploaded workflow file attached to this import
    let filePart: FileUIPart | undefined;
    const storedFile = sessionStorage.getItem(COPILOT_PROMPT_KEYS.file);
    if (storedFile) {
      sessionStorage.removeItem(COPILOT_PROMPT_KEYS.file);
      try {
        const { fileId, fileName, mimeType } = JSON.parse(storedFile);
        // Validate fileId is a UUID to prevent path traversal
        const UUID_RE =
          /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
        if (typeof fileId === "string" && UUID_RE.test(fileId)) {
          filePart = {
            type: "file",
            mediaType: mimeType ?? "application/json",
            filename: fileName ?? "workflow.json",
            url: `/api/proxy/api/workspace/files/${fileId}/download`,
          };
        }
      } catch {
        // ignore malformed stored data
      }
    }

    // Clean up query params
    const cleanURL = new URL(window.location.href);
    cleanURL.searchParams.delete("autosubmit");
    cleanURL.searchParams.delete("source");
    window.history.replaceState(
      null,
      "",
      `${cleanURL.pathname}${cleanURL.search}`,
    );
    return { prompt: storedPrompt.trim(), autosubmit, filePart };
  }

  // Fall back to URL hash (e.g. /copilot#prompt=...)
  const hash = window.location.hash;
  if (!hash) return null;

  const hashParams = new URLSearchParams(hash.slice(1));
  const prompt = hashParams.get("prompt");

  if (!prompt || !prompt.trim()) return null;

  // Clean up hash + autosubmit param only (preserve other query params)
  const cleanURL = new URL(window.location.href);
  cleanURL.hash = "";
  cleanURL.searchParams.delete("autosubmit");
  window.history.replaceState(
    null,
    "",
    `${cleanURL.pathname}${cleanURL.search}`,
  );

  return { prompt: prompt.trim(), autosubmit: false };
}

/**
 * Hook that checks for workflow import data in sessionStorage / URL on mount,
 * and submits it when the app stashed it with an autosubmit intent.
 *
 * Extracted from useCopilotPage to keep that hook focused on page-level concerns.
 */
export function useWorkflowImportAutoSubmit({
  onSend,
  setPendingFileParts,
  isSendLocked = false,
}: {
  onSend: (message: string) => Promise<unknown>;
  setPendingFileParts: (parts: FileUIPart[]) => void;
  isSendLocked?: boolean;
}) {
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  const hasProcessedUrlPrompt = useRef(false);

  // After every commit, not just on mount: a workflow import can stash its
  // prompt once this host is already up, and a later render picks it up.
  useEffect(() => {
    if (isSendLocked || hasProcessedUrlPrompt.current) return;

    const urlPrompt = extractPromptFromUrl();
    if (!urlPrompt) return;

    hasProcessedUrlPrompt.current = true;

    if (urlPrompt.autosubmit) {
      if (urlPrompt.filePart) {
        setPendingFileParts([urlPrompt.filePart]);
      }
      void onSend(urlPrompt.prompt).catch(() => {
        setInitialPrompt(urlPrompt.prompt);
      });
    } else {
      setInitialPrompt(urlPrompt.prompt);
    }
  });
}
