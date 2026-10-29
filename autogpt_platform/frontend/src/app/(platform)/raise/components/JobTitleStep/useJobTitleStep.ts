import type { FormEvent } from "react";
import { useState } from "react";

export const JOB_TITLE_MAX_LENGTH = 100;

interface Args {
  onSubmit: (jobTitle: string) => void;
}

export function useJobTitleStep({ onSubmit }: Args) {
  const [custom, setCustom] = useState("");
  const trimmed = custom.trim();

  function submitCustom(event: FormEvent) {
    event.preventDefault();
    if (trimmed) onSubmit(trimmed);
  }

  return { custom, setCustom, trimmed, submitCustom };
}
