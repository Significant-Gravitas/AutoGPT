import type { FileUIPart } from "ai";
import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "./store";

/**
 * Extract a prompt from the URL hash fragment.
 * Supports: /copilot#prompt=URL-encoded-text
 * Optionally auto-submits if ?autosubmit=true is in the query string.
 * Returns null if no prompt is present.
 */
function extractPromptFromUrl(): {
  prompt: string;
  autosubmit: boolean;
  filePart?: FileUIPart;
} | null {
  if (typeof window === "undefined") return null;

  const searchParams = new URLSearchParams(window.location.search);
  const autosubmit = searchParams.get("autosubmit") === "true";

  // Check sessionStorage first (used by workflow import for large prompts)
  const storedPrompt = sessionStorage.getItem("importWorkflowPrompt");
  if (storedPrompt) {
    sessionStorage.removeItem("importWorkflowPrompt");

    // Check for a pre-uploaded workflow file attached to this import
    let filePart: FileUIPart | undefined;
    const storedFile = sessionStorage.getItem("importWorkflowFile");
    if (storedFile) {
      sessionStorage.removeItem("importWorkflowFile");
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

  return { prompt: prompt.trim(), autosubmit };
}

/**
 * Hook that checks for workflow import data in sessionStorage / URL on mount,
 * and auto-submits a new CoPilot session when `autosubmit=true`.
 *
 * Extracted from useCopilotPage to keep that hook focused on page-level concerns.
 */
export function useWorkflowImportAutoSubmit({
  createSession,
  setPendingMessage,
  pendingFilePartsRef,
}: {
  createSession: () => Promise<string | undefined>;
  setPendingMessage: (msg: string | null) => void;
  pendingFilePartsRef: React.MutableRefObject<FileUIPart[]>;
}) {
  const { setInitialPrompt } = useCopilotUIStore();
  const hasProcessedUrlPrompt = useRef(false);

  useEffect(() => {
    if (hasProcessedUrlPrompt.current) return;

    const urlPrompt = extractPromptFromUrl();
    if (!urlPrompt) return;

    hasProcessedUrlPrompt.current = true;

    if (urlPrompt.autosubmit) {
      if (urlPrompt.filePart) {
        pendingFilePartsRef.current = [urlPrompt.filePart];
      }
      setPendingMessage(urlPrompt.prompt);
      void createSession().catch(() => {
        setPendingMessage(null);
        setInitialPrompt(urlPrompt.prompt);
      });
    } else {
      setInitialPrompt(urlPrompt.prompt);
    }
  }, [createSession, setInitialPrompt, setPendingMessage, pendingFilePartsRef]);
}
