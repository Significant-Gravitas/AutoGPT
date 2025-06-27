"use client";
import * as RXDialog from "@radix-ui/react-dialog";
import { CSSProperties, PropsWithChildren } from "react";
import { Drawer } from "vaul";

import { BaseContent } from "./components/BaseContent";
import { BaseFooter } from "./components/BaseFooter";
import { BaseTrigger } from "./components/BaseTrigger";
import { DialogCtx, useDialogCtx } from "./useDialogCtx";
import { useDialogInternal } from "./useDialogInternal";

interface Props extends PropsWithChildren {
  title?: React.ReactNode;
  styling?: CSSProperties;

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
  styling,

  forceOpen = false,
  onClose,
  controlled,
}: Props) {
  const config = useDialogInternal({ controlled });
  const isOpen = forceOpen || config.isOpen;

  return (
    <DialogCtx.Provider
      value={{
        title: title || "",
        styling,

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
        <RXDialog.Root
          open={isOpen}
          onOpenChange={(open) => {
            if (!open) {
              config.handleClose();
              onClose?.();
            }
          }}
        >
          {children}
        </RXDialog.Root>
      ) : (
        <Drawer.Root
          shouldScaleBackground
          open={isOpen}
          onOpenChange={(open) => {
            if (!open) {
              config.handleClose();
              onClose?.();
            }
          }}
        >
          {children}
        </Drawer.Root>
      )}
    </DialogCtx.Provider>
  );
}

export { Dialog, useDialogCtx };
