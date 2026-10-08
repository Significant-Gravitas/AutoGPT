"use client";

import { Dialog, DialogContent, DialogTitle } from "@/components/ui/dialog";
import { PropsWithChildren } from "react";

interface Props extends PropsWithChildren {
  title: string;
  onClose: () => void;
}

export function FullscreenDialog({ title, onClose, children }: Props) {
  return (
    <Dialog
      open
      onOpenChange={(open) => {
        if (!open) onClose();
      }}
    >
      <DialogContent
        showCloseButton={false}
        className="inset-0 top-0 left-0 flex h-full max-h-none w-full max-w-none translate-x-0 translate-y-0 flex-col gap-0 rounded-none bg-background p-0 shadow-none ring-0 sm:max-w-none"
      >
        <DialogTitle className="sr-only">{title}</DialogTitle>
        {children}
      </DialogContent>
    </Dialog>
  );
}
