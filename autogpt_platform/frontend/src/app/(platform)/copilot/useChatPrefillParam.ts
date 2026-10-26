"use client";

import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useRef } from "react";
import { useCopilotUIStore } from "./store";

/** `/copilot?sessionId=…&prefill=…` opens a thread with a drafted message
 *  in its composer (an answer written for a teammate elsewhere), then drops
 *  the param so a reload doesn't draft it again. */
export function useChatPrefillParam() {
  const [prefill, setPrefill] = useQueryState("prefill", parseAsString);
  const setInitialPrompt = useCopilotUIStore((s) => s.setInitialPrompt);
  // Drafted once per value: clearing the param is async, and the draft must
  // not be re-applied by every render in between.
  const draftedRef = useRef<string | null>(null);
  useEffect(
    function draftPrefill() {
      if (!prefill || draftedRef.current === prefill) return;
      draftedRef.current = prefill;
      setInitialPrompt(prefill);
      void setPrefill(null, { history: "replace" });
    },
    [prefill, setInitialPrompt, setPrefill],
  );
}
