import { useEffect, useRef } from "react";

interface Args {
  prompt: string | null;
  ready: boolean;
  onSend: (prompt: string) => Promise<void>;
  onSettled: () => void;
}

/**
 * Auto-sends a seeded first prompt (`/copilot?seed=...`) exactly once, as
 * soon as the page is ready to open a fresh session. Much lighter than the
 * expert-kickoff flow: seeds are user-clicked navigations, so there is no
 * cross-tab dedupe or retry — just a fire-once guard.
 */
export function useSeededPrompt({ prompt, ready, onSend, onSettled }: Args) {
  const firedRef = useRef(false);
  const onSendRef = useRef(onSend);
  const onSettledRef = useRef(onSettled);

  useEffect(() => {
    onSendRef.current = onSend;
    onSettledRef.current = onSettled;
  }, [onSend, onSettled]);

  useEffect(() => {
    if (!prompt || !ready || firedRef.current) return;
    firedRef.current = true;
    void onSendRef.current(prompt).finally(() => onSettledRef.current());
  }, [prompt, ready]);
}
