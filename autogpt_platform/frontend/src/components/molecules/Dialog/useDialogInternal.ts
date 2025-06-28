import { isLargeScreen, useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { useEffect, useState } from "react";

type Args = {
  controlled:
    | {
        isOpen: boolean;
        set: (open: boolean) => Promise<void> | void;
      }
    | undefined;
};

export function useDialogInternal({ controlled }: Args) {
  const [isOpen, setIsOpen] = useState(false);

  const breakpoint = useBreakpoint();

  const [isLgScreenUp, setIsLgScreenUp] = useState(isLargeScreen(breakpoint));

  useEffect(() => {
    setIsLgScreenUp(isLargeScreen(breakpoint));
  }, [breakpoint]);

  // if first opened as modal, or drawer - we need to keep it this way
  // because, given the current implementation, we can't switch between modal and drawer without a full remount

  async function handleOpen() {
    setIsLgScreenUp(isLargeScreen(breakpoint));

    if (controlled) {
      await controlled.set(true);
    } else {
      setIsOpen(true);
    }
  }

  async function handleClose() {
    if (controlled) {
      await controlled.set(false);
    } else {
      setIsOpen(false);
    }
  }

  return {
    isOpen: controlled ? controlled.isOpen : isOpen,
    handleOpen,
    handleClose,
    isLgScreenUp,
  };
}
