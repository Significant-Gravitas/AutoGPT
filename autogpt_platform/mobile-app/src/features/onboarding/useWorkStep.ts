import { useRef, useState } from "react";

import { useDictation } from "@/features/chat/useDictation";

import { canSendTyped, type WorkAnswer } from "./helpers";

export type WorkScreen = "rest" | "listening" | "typing";

interface Args {
  onSubmit: (answer: WorkAnswer) => void;
  onBackToRole: () => void;
}

/**
 * Talk, type or skip. Spoken words go through the app's dictation and are
 * sent as a typed brain dump; the audio itself is not uploaded.
 */
export function useWorkStep({ onSubmit, onBackToRole }: Args) {
  const [base, setBase] = useState<"rest" | "typing">("rest");
  const [typed, setTyped] = useState("");
  // Only right after a talk attempt: the reason it ended without words.
  const [showNotice, setShowNotice] = useState(false);
  // Between "Start over"'s stop and the new recording, still listening.
  const [isRestarting, setIsRestarting] = useState(false);
  const isStarting = useRef(false);
  const dictation = useDictation({
    getDraft: () => "",
    onText: (text) => onSubmit({ mode: "typed", text: text.trim() }),
  });

  const screen: WorkScreen =
    dictation.isActive || isRestarting ? "listening" : base;

  async function talk() {
    // A second tap while the microphone prompt is up must not leave the screen.
    if (isStarting.current) return;
    isStarting.current = true;
    try {
      await dictation.start();
    } finally {
      isStarting.current = false;
    }
    // Recording now shows "listening". An attempt that ends without words
    // (microphone refused, nothing caught) lands on typing, with the reason.
    setBase("typing");
    setShowNotice(true);
  }

  async function startOver() {
    setIsRestarting(true);
    try {
      await dictation.cancel();
      await dictation.start();
    } finally {
      setIsRestarting(false);
    }
  }

  function sendSpoken() {
    // The words arrive in onText, as they do after the two-minute auto stop.
    void dictation.confirm();
  }

  function write() {
    setShowNotice(false);
    setBase("typing");
  }

  function sendTyped() {
    if (canSendTyped(typed)) onSubmit({ mode: "typed", text: typed.trim() });
  }

  function skip() {
    onSubmit({ mode: "skipped" });
  }

  function back() {
    if (screen === "listening") {
      void dictation.cancel();
      setBase("rest");
    } else if (screen === "typing") {
      setShowNotice(false);
      setBase("rest");
    } else {
      onBackToRole();
    }
  }

  return {
    screen,
    typed,
    setTyped,
    notice: screen === "typing" && showNotice ? dictation.error : null,
    isTranscribing: dictation.isTranscribing,
    recorder: dictation.recorder,
    talk,
    startOver,
    sendSpoken,
    write,
    sendTyped,
    skip,
    back,
  };
}
