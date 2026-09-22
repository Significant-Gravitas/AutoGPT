import type { FormEvent } from "react";
import { useState } from "react";
import { isValidCustomRole, normalizeCustomRole } from "./helpers";

interface Args {
  onPick: (roleId: string) => void;
}

export function useRoleStep({ onPick }: Args) {
  const [custom, setCustom] = useState("");
  const trimmed = normalizeCustomRole(custom);

  function submitCustom(event: FormEvent) {
    event.preventDefault();
    if (!isValidCustomRole(custom)) return;
    onPick(trimmed);
  }

  return { custom, setCustom, trimmed, submitCustom };
}
