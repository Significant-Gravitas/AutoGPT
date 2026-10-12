import { useRef, useState, type MouseEvent } from "react";

export function useDeleteAccountFlowOpener() {
  const [isFlowOpen, setIsFlowOpen] = useState(false);
  const openerRef = useRef<HTMLButtonElement | null>(null);

  function openFlow(event: MouseEvent<HTMLButtonElement>) {
    openerRef.current = event.currentTarget;
    setIsFlowOpen(true);
  }

  function handleFlowOpenChange(open: boolean) {
    setIsFlowOpen(open);
    if (open) return;
    // The Dialog molecule has no Radix trigger, so Radix hands focus back to
    // nothing on close. Return it to the button once the dialog has let go.
    setTimeout(() => openerRef.current?.focus());
  }

  return { isFlowOpen, openFlow, handleFlowOpenChange };
}
