"use client";
import {
  DialogClose,
  DialogContent,
  DialogDescription,
  Dialog as DialogRoot,
  DialogTitle,
} from "@/components/ui/dialog";
import { Drawer } from "@/components/ui/drawer";
import { CSSProperties, PropsWithChildren } from "react";

import { BaseContent } from "./components/BaseContent";
import { BaseFooter } from "./components/BaseFooter";
import { BaseTrigger } from "./components/BaseTrigger";
import { isComposingEscape, isPickerInteraction } from "./helpers";
import { DialogCtx, DialogVariant, useDialogCtx } from "./useDialogCtx";
import { useDialogInternal } from "./useDialogInternal";

interface Props extends PropsWithChildren {
  title?: React.ReactNode;
  /** Linked to the dialog as its accessible description. */
  description?: React.ReactNode;
  /** Keeps the description for screen readers only. */
  hideDescription?: boolean;
  /** `compact` is the dense neutral style: smaller radius, tighter padding,
   *  sans title. */
  variant?: DialogVariant;
  styling?: CSSProperties;
  className?: string;

  forceOpen?: boolean;
  onClose?: (() => void) | undefined;
  controlled?: {
    isOpen: boolean;
    set: (open: boolean) => Promise<void> | void;
  };
}

Dialog.Trigger = BaseTrigger;
Dialog.Content = BaseContent;
Dialog.Footer = BaseFooter;

function Dialog({
  children,
  title,
  description,
  hideDescription = false,
  variant = "default",
  styling,
  className,

  forceOpen = false,
  onClose,
  controlled,
}: Props) {
  const config = useDialogInternal({ controlled });
  const isOpen = forceOpen || config.isOpen;

  function close() {
    config.handleClose();
    onClose?.();
  }

  return (
    <DialogCtx.Provider
      value={{
        title: title || "",
        description,
        hideDescription,
        variant,
        styling,
        className,

        isOpen,
        isForceOpen: forceOpen,
        isLargeScreen: config.isLgScreenUp,
        handleOpen: config.handleOpen,
        handleClose: async () => {
          await config.handleClose();
          onClose?.();
        },
      }}
    >
      {config.isLgScreenUp ? (
        <DialogRoot
          open={isOpen}
          onOpenChange={(open, details) => {
            if (open) return;
            if (
              forceOpen ||
              isComposingEscape(details) ||
              isPickerInteraction(details)
            ) {
              details.cancel();
              return;
            }
            close();
          }}
        >
          {children}
        </DialogRoot>
      ) : (
        <Drawer
          open={isOpen}
          onOpenChange={(open) => {
            if (!open && !forceOpen) close();
          }}
        >
          {children}
        </Drawer>
      )}
    </DialogCtx.Provider>
  );
}

export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogRoot,
  DialogTitle,
  useDialogCtx,
};
