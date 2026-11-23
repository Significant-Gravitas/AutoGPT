// The copilot page sends a stashed prompt on its own, so a send is asked for
// through sessionStorage, which only this origin can write. A prompt that
// arrives in a URL is prefilled instead, for the user to send.

const PROMPT_KEY = "importWorkflowPrompt";
const AUTOSUBMIT_KEY = "importWorkflowAutosubmit";
const FILE_KEY = "importWorkflowFile";

export const COPILOT_PROMPT_KEYS = {
  prompt: PROMPT_KEY,
  autosubmit: AUTOSUBMIT_KEY,
  file: FILE_KEY,
} as const;

interface StashedWorkflowFile {
  fileId: string;
  fileName: string;
  mimeType: string;
}

export function stashCopilotPrompt({
  prompt,
  autosubmit,
  file,
}: {
  prompt: string;
  autosubmit: boolean;
  file?: StashedWorkflowFile;
}) {
  sessionStorage.setItem(PROMPT_KEY, prompt);
  if (autosubmit) {
    sessionStorage.setItem(AUTOSUBMIT_KEY, "true");
  } else {
    sessionStorage.removeItem(AUTOSUBMIT_KEY);
  }
  if (file) {
    sessionStorage.setItem(FILE_KEY, JSON.stringify(file));
  }
}
