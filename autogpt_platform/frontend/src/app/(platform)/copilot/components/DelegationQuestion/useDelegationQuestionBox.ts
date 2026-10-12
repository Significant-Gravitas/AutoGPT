"use client";

import { useState } from "react";
import { isKey } from "@/lib/keyboard";
import { composeAnswer } from "./helpers";

export function useDelegationQuestionBox(onSend: (answer: string) => void) {
  const [picked, setPicked] = useState<string | null>(null);
  const [text, setText] = useState("");
  const answer = composeAnswer(picked, text);

  function pick(option: string) {
    setPicked((current) => (current === option ? null : option));
  }

  function send() {
    if (!answer) return;
    onSend(answer);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLElement>) {
    if (!isKey(e, "Enter") || e.shiftKey) return;
    e.preventDefault();
    send();
  }

  return { picked, pick, text, setText, answer, send, handleKeyDown };
}
