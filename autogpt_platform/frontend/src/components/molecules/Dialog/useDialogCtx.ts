import { CSSProperties, createContext, useContext } from "react";

export function useDialogCtx() {
  const modalContext = useContext(DialogCtx);

  return modalContext;
}

export interface DialogCtx {
  title: React.ReactNode;
  handleOpen: () => void;
  handleClose: () => void;
  isOpen: boolean;
  isForceOpen: boolean;
  isLargeScreen: boolean;
  styling: CSSProperties | undefined;
}

export const DialogCtx = createContext<DialogCtx>({
  title: "",
  isOpen: false,
  isForceOpen: false,
  isLargeScreen: true,
  handleOpen: () => undefined,
  handleClose: () => undefined,
  styling: {},
});
