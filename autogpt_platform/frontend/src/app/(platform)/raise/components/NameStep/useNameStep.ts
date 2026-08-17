import type { FormEvent } from "react";
import { useState } from "react";

interface Args {
  onSubmit: (name: string) => void;
}

export function useNameStep({ onSubmit }: Args) {
  const [custom, setCustom] = useState("");
  const trimmed = custom.trim();

  function submitCustom(event: FormEvent) {
    event.preventDefault();
    if (trimmed) onSubmit(trimmed);
  }

  return { custom, setCustom, trimmed, submitCustom };
}
