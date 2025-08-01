import React, { PropsWithChildren } from "react";

import { useDialogCtx } from "../useDialogCtx";
import { cn } from "@/lib/utils";

export function BaseTrigger({ children }: PropsWithChildren) {
  const ctx = useDialogCtx();
  const child = children as React.ReactElement;

  return React.cloneElement(child, {
    onClick: ctx.handleOpen,
    className: cn("cursor-pointer", child.props.className),
  });
}
