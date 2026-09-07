import { CSSProperties, createContext, useContext } from "react";

export function useDialogCtx() {
  const modalContext = useContext(DialogCtx);

  return modalContext;
}

export type DialogVariant = "default" | "compact";

export interface DialogCtx {
  title: React.ReactNode;
  variant: DialogVariant;
  handleOpen: () => void;
  handleClose: () => void;
  isOpen: boolean;
  isForceOpen: boolean;
  isLargeScreen: boolean;
  styling: CSSProperties | undefined;
  className?: string;
}

export const DialogCtx = createContext<DialogCtx>({
  title: "",
  variant: "default",
  isOpen: false,
  isForceOpen: false,
  isLargeScreen: true,
  handleOpen: () => undefined,
  handleClose: () => undefined,
  styling: {},
});
