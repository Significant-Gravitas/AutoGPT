import { PropsWithChildren } from "react";

import { useDialogCtx } from "../useDialogCtx";
import { DialogWrap } from "./DialogWrap";
import { DrawerWrap } from "./DrawerWrap";

type Props = PropsWithChildren;

export function BaseContent({ children }: Props) {
  const ctx = useDialogCtx();

  return ctx.isLargeScreen ? (
    <DialogWrap {...ctx}>{children}</DialogWrap>
  ) : (
    <DrawerWrap {...ctx}>{children}</DrawerWrap>
  );
}
