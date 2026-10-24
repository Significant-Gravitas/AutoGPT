import { useState } from "react";

type Step = "reason" | "confirm";

interface Args {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

const STEP_TITLE: Record<Step, string> = {
  reason: "Why are you leaving?",
  confirm: "Delete your account?",
};

export function useDeleteAccountFlow({ open, onOpenChange }: Args) {
  const [step, setStep] = useState<Step>("reason");
  const [session, setSession] = useState(0);
  const [wasOpen, setWasOpen] = useState(open);
  const [isDeleting, setIsDeleting] = useState(false);

  // Every open starts over at the reason step with nothing picked; the
  // session key remounts the steps so their own state starts over too.
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) {
      setStep("reason");
      setSession((current) => current + 1);
    }
  }

  function handleOpenChange(next: boolean) {
    if (!next && isDeleting) return;
    onOpenChange(next);
  }

  function close() {
    handleOpenChange(false);
  }

  function goToConfirm() {
    setStep("confirm");
  }

  return {
    step,
    session,
    title: STEP_TITLE[step],
    isDeleting,
    setIsDeleting,
    handleOpenChange,
    close,
    goToConfirm,
  };
}
