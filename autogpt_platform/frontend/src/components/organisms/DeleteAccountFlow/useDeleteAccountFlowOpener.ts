import { usePostAccountSendExitFeedback } from "@/app/api/__generated__/endpoints/account/account";
import { useRef, useState, type MouseEvent } from "react";
import type { DeletionSource } from "./helpers";

export function useDeleteAccountFlowOpener(source: DeletionSource) {
  const [isFlowOpen, setIsFlowOpen] = useState(false);
  const openerRef = useRef<HTMLButtonElement | null>(null);
  const { mutate: sendFeedback } = usePostAccountSendExitFeedback();

  function openFlow(event: MouseEvent<HTMLButtonElement>) {
    openerRef.current = event.currentTarget;
    setIsFlowOpen(true);
    // Counts the start, fire and forget: closing with the X sends nothing else.
    sendFeedback({ data: { kind: "opened", surface: source } });
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
