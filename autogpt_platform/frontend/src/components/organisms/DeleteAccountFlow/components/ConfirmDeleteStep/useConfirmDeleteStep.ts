import { useState, type FormEvent } from "react";
import type { DeletionSource } from "../../helpers";
import { emailMatches } from "./helpers";
import { useDeleteAccount } from "./useDeleteAccount";
import { useDeletionConsequences } from "./useDeletionConsequences";

interface Args {
  source: DeletionSource;
  onBusyChange: (busy: boolean) => void;
}

export function useConfirmDeleteStep({ source, onBusyChange }: Args) {
  const [typedEmail, setTypedEmail] = useState("");
  const consequences = useDeletionConsequences();
  const { email, isDeleting, deleteAccount } = useDeleteAccount({
    source,
    onBusyChange,
  });
  const canDelete = !isDeleting && emailMatches({ typed: typedEmail, email });

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!canDelete) return;
    void deleteAccount(typedEmail.trim());
  }

  return {
    email,
    typedEmail,
    setTypedEmail,
    consequences,
    canDelete,
    isDeleting,
    handleSubmit,
  };
}
