import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { cn } from "@/lib/utils";
import type { ReactNode } from "react";
import { useStepFooter } from "./useStepFooter";

interface Props {
  children: ReactNode;
}

export function StepFooter({ children }: Props) {
  const { sentinelRef, hasContentBelow } = useStepFooter();

  return (
    <>
      <Dialog.Footer
        className={cn(
          // Static on very short viewports (400% zoom), where a sticky
          // footer would cover everything above it.
          "sticky bottom-0 z-10 mt-2 flex-wrap gap-3 border-t bg-surface pt-4 [@media(max-height:300px)]:static",
          hasContentBelow ? "border-separator" : "border-transparent",
        )}
      >
        {children}
      </Dialog.Footer>
      <div ref={sentinelRef} aria-hidden className="h-px shrink-0" />
    </>
  );
}
